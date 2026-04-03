/**
 * ucx_sender.c — UCX trace-driven traffic sender (RoCE)
 *
 * Reads a V2 binary rate schedule (one target rate in bytes/sec per
 * window) and paces sends with a token bucket.  Produces continuous,
 * smooth traffic matching the trace's BW pattern — NOT on/off
 * duty-cycling.
 *
 * Token bucket design:
 *   - ns-resolution refill from CLOCK_MONOTONIC
 *   - rate updated per window (no token reset — carryover smooths edges)
 *   - spin-wait with UCX progress interleaving for sub-µs accuracy
 *   - burst limit caps token accumulation (prevents mega-bursts)
 *
 * Build:
 *   gcc -O3 -march=native -o ucx_sender ucx_sender.c \
 *       $(pkg-config --cflags --libs ucx) -lm
 *
 * Usage:
 *   # Rate-based (V2 schedule from interfere.py):
 *   UCX_TLS=rc UCX_NET_DEVICES=mlx5_1:1 \
 *       ./ucx_sender 10.0.0.2 18515 --schedule schedule_v2.bin
 *
 *   # Constant rate:
 *   UCX_TLS=rc UCX_NET_DEVICES=mlx5_1:1 \
 *       ./ucx_sender 10.0.0.2 18515 --rate-bps 5000000000 --duration 300
 */

#define _GNU_SOURCE
#include <ucp/api/ucp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <math.h>
#include <signal.h>
#include <errno.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <getopt.h>

/* ------------------------------------------------------------------ */
/* Tunables                                                            */
/* ------------------------------------------------------------------ */
#define SPIN_THRESHOLD_NS   100000ULL     /* 100µs: spin vs sleep      */
#define MAX_OUTSTANDING     16            /* max in-flight sends        */
#define TAG_DATA            0xBEEFCAFEULL
#define DEFAULT_MSG_SIZE    65536         /* 64 KB for sub-ms precision */
#define DEFAULT_WINDOW_NS   1000000ULL   /* 1 ms                       */
#define TOKEN_BUCKET_BURST  (1024*1024)  /* 1 MB max burst             */
#define RATE_EPSILON        1.0          /* bytes/sec below = idle      */
#define STATS_INTERVAL_SEC  5.0          /* print stats every N sec     */
#define CONNECT_RETRY_SEC   2
#define CONNECT_MAX_RETRIES 15

/* Schedule version */
#define SCHED_VERSION_V2    2            /* rate_bps (double)           */

/* ------------------------------------------------------------------ */
/* Schedule header (32 bytes)                                          */
/* ------------------------------------------------------------------ */
typedef struct __attribute__((packed)) {
    char     magic[4];      /* "SCHD"                              */
    uint32_t version;       /* 2 = rate-based                      */
    uint32_t num_windows;
    uint32_t msg_size;      /* bytes per message                   */
    uint64_t window_ns;
    uint8_t  pad[8];
} sched_header_t;           /* 32 bytes                            */

/* ------------------------------------------------------------------ */
/* Schedule                                                            */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t  num_windows;
    uint64_t  window_ns;
    uint32_t  msg_size;
    double   *rates_bps;    /* bytes/sec per window                 */
} schedule_t;

/* ------------------------------------------------------------------ */
/* Send tracking                                                       */
/* ------------------------------------------------------------------ */
typedef struct {
    volatile int completed;
    ucs_status_t status;
} send_ctx_t;

/* ------------------------------------------------------------------ */
/* Token bucket (ns-resolution)                                        */
/* ------------------------------------------------------------------ */
typedef struct {
    double   tokens;
    double   rate_bps;      /* bytes/sec for current window         */
    uint64_t last_ns;       /* last refill (CLOCK_MONOTONIC)        */
    double   burst_limit;
} token_bucket_t;

/* ------------------------------------------------------------------ */
/* Globals                                                             */
/* ------------------------------------------------------------------ */
static volatile int g_stop        = 0;
static volatile int g_outstanding = 0;
static volatile int g_sending     = 0;  /* set when send loop starts */
static volatile int g_ep_error    = 0;  /* set by err_cb during settle */

static void sig_handler(int s) { (void)s; g_stop = 1; }

/* ------------------------------------------------------------------ */
/* Clock                                                               */
/* ------------------------------------------------------------------ */
static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
}

static inline void adaptive_wait_until(uint64_t target_ns,
                                       ucp_worker_h worker) {
    uint64_t cur = now_ns();
    if (cur >= target_ns) return;
    uint64_t gap = target_ns - cur;

    if (gap > SPIN_THRESHOLD_NS) {
        uint64_t sleep_ns = gap - SPIN_THRESHOLD_NS;
        struct timespec ts = {
            .tv_sec  = sleep_ns / 1000000000ULL,
            .tv_nsec = sleep_ns % 1000000000ULL,
        };
        nanosleep(&ts, NULL);
    }
    while (now_ns() < target_ns)
        ucp_worker_progress(worker);
}

/* ------------------------------------------------------------------ */
/* Token bucket operations                                             */
/* ------------------------------------------------------------------ */
static void tb_init(token_bucket_t *tb, double rate_bps) {
    tb->tokens      = (double)TOKEN_BUCKET_BURST;
    tb->rate_bps    = rate_bps;
    tb->last_ns     = now_ns();
    tb->burst_limit = (double)TOKEN_BUCKET_BURST;
}

static inline void tb_refill(token_bucket_t *tb) {
    uint64_t now = now_ns();
    double elapsed_s = (double)(now - tb->last_ns) * 1e-9;
    tb->tokens  += elapsed_s * tb->rate_bps;
    if (tb->tokens > tb->burst_limit)
        tb->tokens = tb->burst_limit;
    tb->last_ns  = now;
}

/**
 * Spin-wait (with UCX progress) until enough tokens, then consume.
 * Returns 0 on success, -1 if g_stop or past deadline.
 */
static inline int tb_consume(token_bucket_t *tb, uint64_t bytes,
                             ucp_worker_h worker, uint64_t deadline_ns) {
    while (1) {
        if (g_stop) return -1;

        tb_refill(tb);
        if (tb->tokens >= (double)bytes) {
            tb->tokens -= (double)bytes;
            return 0;
        }

        /* Check deadline */
        uint64_t now = now_ns();
        if (now >= deadline_ns) return -1;

        /* Compute wait time for the deficit */
        double deficit = (double)bytes - tb->tokens;
        uint64_t wait_ns = (uint64_t)(deficit / tb->rate_bps * 1e9);

        /* Cap wait to not exceed deadline */
        uint64_t remain = deadline_ns - now;
        if (wait_ns > remain) wait_ns = remain;

        if (wait_ns > SPIN_THRESHOLD_NS) {
            uint64_t sleep_ns = wait_ns - SPIN_THRESHOLD_NS;
            struct timespec ts = {
                .tv_sec  = sleep_ns / 1000000000ULL,
                .tv_nsec = sleep_ns % 1000000000ULL,
            };
            nanosleep(&ts, NULL);
        }

        /* Short spin + UCX progress */
        uint64_t spin_deadline = now_ns() +
            (wait_ns < SPIN_THRESHOLD_NS ? wait_ns : SPIN_THRESHOLD_NS);
        while (now_ns() < spin_deadline)
            ucp_worker_progress(worker);
    }
}

/* ------------------------------------------------------------------ */
/* UCX callbacks                                                       */
/* ------------------------------------------------------------------ */
static void send_cb(void *request, ucs_status_t status, void *user_data) {
    (void)status; (void)user_data;
    g_outstanding--;
    ucp_request_free(request);
}

static void req_init(void *request) {
    memset(request, 0, sizeof(send_ctx_t));
}

/* ------------------------------------------------------------------ */
/* Schedule loading                                                    */
/* ------------------------------------------------------------------ */
static int load_schedule_file(const char *path, schedule_t *sched) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("[sender] fopen schedule"); return -1; }

    sched_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "[sender] failed to read schedule header\n");
        fclose(f); return -1;
    }
    if (memcmp(hdr.magic, "SCHD", 4) != 0) {
        fprintf(stderr, "[sender] invalid schedule magic\n");
        fclose(f); return -1;
    }
    if (hdr.version != SCHED_VERSION_V2) {
        fprintf(stderr, "[sender] unsupported schedule version %u "
                "(expected V2=%d)\n", hdr.version, SCHED_VERSION_V2);
        fclose(f); return -1;
    }

    sched->num_windows = hdr.num_windows;
    sched->window_ns   = hdr.window_ns;
    sched->msg_size    = hdr.msg_size;

    sched->rates_bps = malloc(hdr.num_windows * sizeof(double));
    if (!sched->rates_bps) { fclose(f); return -1; }
    size_t rd = fread(sched->rates_bps, sizeof(double),
                      hdr.num_windows, f);
    fclose(f);
    if (rd != hdr.num_windows) {
        fprintf(stderr, "[sender] short read: %zu / %u doubles\n",
                rd, hdr.num_windows);
        free(sched->rates_bps); return -1;
    }

    /* Stats */
    double sum = 0, max_rate = 0;
    uint32_t active = 0;
    for (uint32_t i = 0; i < sched->num_windows; i++) {
        double r = sched->rates_bps[i];
        if (r < 0) sched->rates_bps[i] = 0;
        if (r > RATE_EPSILON) active++;
        sum += r;
        if (r > max_rate) max_rate = r;
    }
    double avg = sched->num_windows > 0 ?
                 sum / sched->num_windows : 0;
    double dur = (double)sched->num_windows *
                 sched->window_ns / 1e9;

    printf("[sender] schedule: %u windows × %.2f ms = %.1f s, "
           "msg=%u B, %u active (%.1f%%), "
           "avg=%.1f Gbps, peak=%.1f Gbps\n",
           sched->num_windows, sched->window_ns / 1e6, dur,
           sched->msg_size, active,
           active * 100.0 / sched->num_windows,
           avg * 8.0 / 1e9, max_rate * 8.0 / 1e9);

    return 0;
}

static void make_constant_rate_schedule(schedule_t *sched, double rate_bps,
                                        uint64_t window_ns, uint32_t msg_size,
                                        double duration_sec) {
    uint32_t n = (uint32_t)ceil(duration_sec / (window_ns / 1e9));
    if (n == 0) n = 1;

    sched->num_windows = n;
    sched->window_ns   = window_ns;
    sched->msg_size    = msg_size;
    sched->rates_bps   = malloc(n * sizeof(double));
    for (uint32_t i = 0; i < n; i++)
        sched->rates_bps[i] = rate_bps;

    printf("[sender] constant rate: %.1f Gbps, %u windows × %.3f ms\n",
           rate_bps * 8.0 / 1e9, n, window_ns / 1e6);
}

/* ------------------------------------------------------------------ */
/* UCX: connect to receiver                                            */
/* ------------------------------------------------------------------ */
static void sender_ep_err_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
    (void)arg; (void)ep;
    if (g_sending) {
        fprintf(stderr, "[sender] endpoint error: %s\n",
                ucs_status_string(status));
        g_stop = 1;
    } else {
        fprintf(stderr, "[sender] stale EP error during settle: %s (will reconnect)\n",
                ucs_status_string(status));
        g_ep_error = 1;
    }
}

static ucp_ep_h connect_to_server(ucp_worker_h worker,
                                  const char *addr, uint16_t port) {
    struct sockaddr_in sa = { 0 };
    sa.sin_family = AF_INET;
    sa.sin_port   = htons(port);
    inet_pton(AF_INET, addr, &sa.sin_addr);

    ucp_ep_params_t ep_params = {
        .field_mask       = UCP_EP_PARAM_FIELD_FLAGS |
                            UCP_EP_PARAM_FIELD_SOCK_ADDR |
                            UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                            UCP_EP_PARAM_FIELD_ERR_HANDLER,
        .flags            = UCP_EP_PARAMS_FLAGS_CLIENT_SERVER,
        .sockaddr.addr    = (struct sockaddr *)&sa,
        .sockaddr.addrlen = sizeof(sa),
        .err_mode         = UCP_ERR_HANDLING_MODE_PEER,
        .err_handler      = { .cb = sender_ep_err_cb, .arg = NULL },
    };
    ucp_ep_h ep;
    ucs_status_t st = ucp_ep_create(worker, &ep_params, &ep);
    if (st != UCS_OK) {
        fprintf(stderr, "[sender] ucp_ep_create: %s\n",
                ucs_status_string(st));
        return NULL;
    }
    return ep;
}

/* ------------------------------------------------------------------ */
/* Post one non-blocking send                                          */
/* Returns: 0 success, -1 error                                       */
/* ------------------------------------------------------------------ */
static inline int post_send(ucp_ep_h ep, void *buf, uint32_t msg_size,
                            uint64_t *total_bytes, uint64_t *total_msgs,
                            uint64_t *stats_bytes) {
    ucp_request_param_t rp = {
        .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK,
        .cb.send      = send_cb,
    };
    ucs_status_ptr_t ret = ucp_tag_send_nbx(
        ep, buf, msg_size, TAG_DATA, &rp);

    if (UCS_PTR_IS_ERR(ret))
        return -1;

    if (ret != NULL)
        g_outstanding++;

    *total_bytes += msg_size;
    *total_msgs  += 1;
    *stats_bytes += msg_size;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Rate-shaped send loop (token-bucket paced)                          */
/* ------------------------------------------------------------------ */
static void run_rate_loop(schedule_t *sched, ucp_ep_h ep,
                          ucp_worker_h worker, void *buf,
                          uint64_t total_windows) {
    uint32_t msg_size = sched->msg_size;
    token_bucket_t tb;
    tb_init(&tb, sched->rates_bps[0]);

    uint64_t total_bytes = 0, total_msgs = 0;
    uint64_t start_time  = now_ns();
    uint64_t last_stats  = start_time;
    uint64_t stats_bytes = 0;
    uint64_t send_errors = 0;
    uint64_t idle_windows = 0;

    printf("[sender] rate loop: msg_size=%u, window=%.2f ms, "
           "%llu windows\n",
           msg_size, (double)sched->window_ns / 1e6,
           (unsigned long long)total_windows);

    for (uint64_t w = 0; w < total_windows && !g_stop; w++) {
        double rate = sched->rates_bps[w % sched->num_windows];
        uint64_t window_start = start_time + w * sched->window_ns;
        uint64_t window_end   = window_start + sched->window_ns;

        /* Wait for window start */
        adaptive_wait_until(window_start, worker);

        /* Update token bucket rate (do NOT reset tokens) */
        tb.rate_bps = rate;

        /* Idle window: just progress */
        if (rate < RATE_EPSILON) {
            idle_windows++;
            adaptive_wait_until(window_end, worker);
            goto stats_check;
        }

        /* Rate-shaped sending within the window */
        while (now_ns() < window_end && !g_stop) {
            /* Pipeline management */
            while (g_outstanding >= MAX_OUTSTANDING && !g_stop)
                ucp_worker_progress(worker);
            if (g_stop) break;

            /* Token bucket: wait for tokens */
            if (tb_consume(&tb, msg_size, worker, window_end) < 0)
                break;  /* window ended or g_stop */

            /* Post send */
            if (post_send(ep, buf, msg_size,
                          &total_bytes, &total_msgs, &stats_bytes) < 0) {
                send_errors++;
                if (send_errors > 100) { g_stop = 1; break; }
            }

            ucp_worker_progress(worker);
        }

    stats_check:
        ;   /* Periodic stats */
        uint64_t now = now_ns();
        double since_stats = (double)(now - last_stats) / 1e9;
        if (since_stats >= STATS_INTERVAL_SEC) {
            double bw_gbps = (stats_bytes * 8.0) / (since_stats * 1e9);
            double elapsed = (double)(now - start_time) / 1e9;
            double cur_rate_gbps = rate * 8.0 / 1e9;
            printf("[sender] t=%.1fs: %.2f Gbps sent "
                   "(target=%.2f Gbps, total=%.2f GB, "
                   "%" PRIu64 " msgs)\n",
                   elapsed, bw_gbps, cur_rate_gbps,
                   total_bytes / 1e9, total_msgs);
            stats_bytes = 0;
            last_stats  = now;
        }
    }

    /* Drain outstanding */
    if (g_outstanding > 0) {
        printf("[sender] draining %d outstanding...\n", g_outstanding);
        uint64_t t0 = now_ns();
        while (g_outstanding > 0 && (now_ns() - t0) < 5000000000ULL)
            ucp_worker_progress(worker);
    }

    /* Summary */
    double total_sec = (double)(now_ns() - start_time) / 1e9;
    printf("\n=== Sender Summary ===\n");
    printf("  Duration      : %.1f sec\n", total_sec);
    printf("  Total sent    : %.2f GB (%" PRIu64 " msgs)\n",
           total_bytes / 1e9, total_msgs);
    printf("  Avg BW        : %.2f Gbps\n",
           total_bytes * 8.0 / (total_sec * 1e9));
    printf("  Msg size      : %u bytes\n", msg_size);
    printf("  Window        : %.2f ms\n", sched->window_ns / 1e6);
    printf("  Idle windows  : %" PRIu64 "\n", idle_windows);
    if (send_errors > 0)
        printf("  Send errors   : %" PRIu64 "\n", send_errors);
}

/* ------------------------------------------------------------------ */
/* Usage                                                               */
/* ------------------------------------------------------------------ */
static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <server_ip> <port> [options]\n\n"
        "Options:\n"
        "  --schedule <file>   Binary V2 rate schedule\n"
        "  --rate-bps <N>      Constant rate in bytes/sec\n"
        "  --duration <sec>    Run duration (default: from schedule)\n"
        "  --msg-size <bytes>  Message size (default: %d)\n"
        "  --window-ns <ns>    Window size in ns (default: %llu)\n",
        prog, DEFAULT_MSG_SIZE, (unsigned long long)DEFAULT_WINDOW_NS);
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    if (argc < 3) { usage(argv[0]); return 1; }

    const char *server_ip = argv[1];
    uint16_t    port      = (uint16_t)atoi(argv[2]);

    const char *schedule_path = NULL;
    double      const_rate    = -1;    /* bytes/sec, -1 = not set     */
    double      duration_sec  = 0;
    uint32_t    msg_size      = DEFAULT_MSG_SIZE;
    uint64_t    window_ns     = DEFAULT_WINDOW_NS;

    static struct option long_opts[] = {
        {"schedule",  required_argument, 0, 's'},
        {"rate-bps",  required_argument, 0, 'r'},
        {"duration",  required_argument, 0, 'D'},
        {"msg-size",  required_argument, 0, 'm'},
        {"window-ns", required_argument, 0, 'w'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    optind = 3;
    int c;
    while ((c = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (c) {
            case 's': schedule_path = optarg; break;
            case 'r': const_rate = strtod(optarg, NULL); break;
            case 'D': duration_sec = strtod(optarg, NULL); break;
            case 'm': msg_size = (uint32_t)atol(optarg); break;
            case 'w': window_ns = (uint64_t)atoll(optarg); break;
            case 'h': usage(argv[0]); return 0;
            default:  usage(argv[0]); return 1;
        }
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    /* ---- Load or create schedule ---- */
    schedule_t sched = { 0 };
    if (schedule_path) {
        if (load_schedule_file(schedule_path, &sched) < 0)
            return 1;
        msg_size  = sched.msg_size;
        window_ns = sched.window_ns;
    } else {
        /* Constant rate (default: full speed via large rate) */
        if (const_rate < 0) const_rate = 100e9;  /* 100 GB/s = ~full speed */
        if (duration_sec <= 0) duration_sec = 3600;
        make_constant_rate_schedule(&sched, const_rate, window_ns,
                                    msg_size, duration_sec);
    }

    /* Total windows */
    double sched_dur = (double)sched.num_windows * sched.window_ns / 1e9;
    if (duration_sec <= 0) duration_sec = sched_dur;
    uint64_t total_windows = (uint64_t)ceil(
        duration_sec * 1e9 / (double)sched.window_ns);

    printf("[sender] duration=%.1f s (%llu windows)%s\n",
           duration_sec, (unsigned long long)total_windows,
           duration_sec > sched_dur ? " (loops)" : "");

    /* ---- UCX init ---- */
    ucp_params_t ucp_params = {
        .field_mask   = UCP_PARAM_FIELD_FEATURES |
                        UCP_PARAM_FIELD_REQUEST_SIZE |
                        UCP_PARAM_FIELD_REQUEST_INIT,
        .features     = UCP_FEATURE_TAG,
        .request_size = sizeof(send_ctx_t),
        .request_init = req_init,
    };
    ucp_context_h context;
    ucs_status_t st = ucp_init(&ucp_params, NULL, &context);
    if (st != UCS_OK) {
        fprintf(stderr, "[sender] ucp_init: %s\n", ucs_status_string(st));
        return 1;
    }

    ucp_worker_params_t wparams = {
        .field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCS_THREAD_MODE_SINGLE,
    };
    ucp_worker_h worker;
    st = ucp_worker_create(context, &wparams, &worker);
    if (st != UCS_OK) {
        fprintf(stderr, "[sender] ucp_worker_create: %s\n",
                ucs_status_string(st));
        return 1;
    }

    /* Connect with retry — if a stale error breaks the EP during settle,
     * close it and reconnect with a fresh EP */
    ucp_ep_h ep = NULL;
    for (int attempt = 0; attempt < CONNECT_MAX_RETRIES; attempt++) {
        printf("[sender] connecting to %s:%u (attempt %d)...\n",
               server_ip, port, attempt + 1);
        g_ep_error = 0;
        ep = connect_to_server(worker, server_ip, port);
        if (!ep) {
            if (g_stop) goto cleanup_ucx;
            sleep(CONNECT_RETRY_SEC);
            continue;
        }

        /* Settle: progress UCX to flush stale errors */
        printf("[sender] settling connection...\n");
        {
            uint64_t settle_end = now_ns() + 1000000000ULL;  /* 1s settle */
            while (now_ns() < settle_end && !g_stop && !g_ep_error)
                ucp_worker_progress(worker);
        }

        if (g_ep_error) {
            printf("[sender] EP error during settle, closing and retrying...\n");
            ucp_request_param_t cp = {
                .op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS,
                .flags        = UCP_EP_CLOSE_FLAG_FORCE,
            };
            ucs_status_ptr_t cr = ucp_ep_close_nbx(ep, &cp);
            if (cr != NULL && !UCS_PTR_IS_ERR(cr)) {
                uint64_t t0 = now_ns();
                while (ucp_request_check_status(cr) == UCS_INPROGRESS &&
                       (now_ns() - t0) < 3000000000ULL)
                    ucp_worker_progress(worker);
                ucp_request_free(cr);
            }
            ep = NULL;
            sleep(CONNECT_RETRY_SEC);
            continue;
        }

        printf("[sender] connected\n");
        break;
    }
    if (!ep) {
        fprintf(stderr, "[sender] failed to connect after %d attempts\n",
                CONNECT_MAX_RETRIES);
        goto cleanup_ucx;
    }

    /* ---- Send buffer ---- */
    void *buf = malloc(msg_size);
    if (!buf) { fprintf(stderr, "[sender] malloc failed\n"); goto cleanup_ep; }
    memset(buf, 0xAB, msg_size);

    /* ---- Run ---- */
    g_sending = 1;
    run_rate_loop(&sched, ep, worker, buf, total_windows);
    g_sending = 0;

    /* ---- Cleanup ---- */
    free(buf);

cleanup_ep:
    /* Close EP: use force-close to avoid blocking on errored EP */
    if (ep) {
        ucp_request_param_t close_params = {
            .op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS,
            .flags        = UCP_EP_CLOSE_FLAG_FORCE,
        };
        ucs_status_ptr_t close_req = ucp_ep_close_nbx(ep, &close_params);
        if (close_req != NULL && !UCS_PTR_IS_ERR(close_req)) {
            uint64_t t0 = now_ns();
            while (ucp_request_check_status(close_req) == UCS_INPROGRESS &&
                   (now_ns() - t0) < 3000000000ULL)
                ucp_worker_progress(worker);
            ucp_request_free(close_req);
        }
    }
cleanup_ucx:
    ucp_worker_destroy(worker);
    ucp_cleanup(context);
    free(sched.rates_bps);
    return 0;
}

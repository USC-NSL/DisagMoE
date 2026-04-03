/**
 * ucx_ring_node.c — Combined UCX sender+receiver in a single process
 *
 * Uses ONE UCX context and ONE worker to avoid UD QP conflicts that
 * arise when two separate UCX processes run on the same node.
 *
 * If --send-ip is omitted, runs receiver-only.
 * If --recv-port is omitted, runs sender-only.
 *
 * Build:
 *   gcc -O3 -march=native -o ucx_ring_node ucx_ring_node.c \
 *       $(pkg-config --cflags --libs ucx) -lm
 *
 * Usage:
 *   UCX_TLS=rc UCX_NET_DEVICES=mlx5_1:1 \
 *       ./ucx_ring_node --send-ip 10.0.0.2 --send-port 18515 \
 *                       --recv-port 18518 \
 *                       --schedule schedule.bin \
 *                       [--duration 300] [--msg-size 65536] [--csv recv_bw.csv]
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
#define SPIN_THRESHOLD_NS    100000ULL        /* 100 us                */
#define MAX_SEND_OUTSTANDING 16               /* max in-flight sends   */
#define MAX_RECV_OUTSTANDING 256              /* pipelined receives    */
#define TAG_DATA             0xBEEFCAFEULL
#define TAG_MASK             UINT64_MAX
#define DEFAULT_MSG_SIZE     65536            /* 64 KB                 */
#define DEFAULT_WINDOW_NS    1000000ULL       /* 1 ms                  */
#define DEFAULT_MAX_RECV_MSG (16 * 1024 * 1024) /* 16 MB              */
#define TOKEN_BUCKET_BURST   (1024*1024)      /* 1 MB max burst        */
#define RATE_EPSILON         1.0              /* bytes/sec below=idle  */
#define STATS_INTERVAL_SEC   5.0
#define CONNECT_RETRY_SEC    3
#define CONNECT_MAX_RETRIES  30
#define RECV_WINDOW_NS       1000000ULL       /* 1 ms BW tracking      */
#define RECV_PRINT_EVERY_N   5000             /* print every N windows */

/* Schedule version */
#define SCHED_VERSION_V2     2

/* ------------------------------------------------------------------ */
/* Schedule header (32 bytes)                                          */
/* ------------------------------------------------------------------ */
typedef struct __attribute__((packed)) {
    char     magic[4];
    uint32_t version;
    uint32_t num_windows;
    uint32_t msg_size;
    uint64_t window_ns;
    uint8_t  pad[8];
} sched_header_t;

/* ------------------------------------------------------------------ */
/* Schedule                                                            */
/* ------------------------------------------------------------------ */
typedef struct {
    uint32_t  num_windows;
    uint64_t  window_ns;
    uint32_t  msg_size;
    double   *rates_bps;
} schedule_t;

/* ------------------------------------------------------------------ */
/* UCX request contexts                                                */
/* ------------------------------------------------------------------ */
typedef struct {
    volatile int completed;
    ucs_status_t status;
} send_ctx_t;

typedef struct {
    volatile int completed;
    ucs_status_t status;
    size_t       length;
} recv_ctx_t;

/* Use the larger of the two for request_size */
#define REQ_SIZE  (sizeof(recv_ctx_t) > sizeof(send_ctx_t) ? \
                   sizeof(recv_ctx_t) : sizeof(send_ctx_t))

/* ------------------------------------------------------------------ */
/* Token bucket                                                        */
/* ------------------------------------------------------------------ */
typedef struct {
    double   tokens;
    double   rate_bps;
    uint64_t last_ns;
    double   burst_limit;
} token_bucket_t;

/* ------------------------------------------------------------------ */
/* Receive slot                                                        */
/* ------------------------------------------------------------------ */
typedef struct {
    void            *buf;
    recv_ctx_t       ctx;
    ucs_status_ptr_t req;
    int              active;
} recv_slot_t;

/* ------------------------------------------------------------------ */
/* Globals                                                             */
/* ------------------------------------------------------------------ */
static volatile int g_stop           = 0;  /* global stop (Ctrl+C) */
static volatile int g_send_stop      = 0;  /* sender-side error */
static volatile int g_recv_stop      = 0;  /* receiver-side error */
static volatile int g_recv_new_client = 0; /* new client connected */
static volatile int g_send_outstanding = 0;
static volatile int g_sending        = 0;
static volatile int g_receiving      = 0;
static volatile int g_ep_error       = 0;
static ucp_ep_h     g_recv_client_ep = NULL;

static void sig_handler(int s) { (void)s; g_stop = 1; }

/* ------------------------------------------------------------------ */
/* Clock                                                               */
/* ------------------------------------------------------------------ */
static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + (uint64_t)ts.tv_nsec;
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
    tb->tokens += elapsed_s * tb->rate_bps;
    if (tb->tokens > tb->burst_limit)
        tb->tokens = tb->burst_limit;
    tb->last_ns = now;
}

/**
 * Non-blocking token check: refill and try to consume.
 * Returns 0 if consumed, -1 if not enough tokens (caller should retry later).
 */
static inline int tb_try_consume(token_bucket_t *tb, uint64_t bytes) {
    tb_refill(tb);
    if (tb->tokens >= (double)bytes) {
        tb->tokens -= (double)bytes;
        return 0;
    }
    return -1;
}

/* ------------------------------------------------------------------ */
/* UCX callbacks                                                       */
/* ------------------------------------------------------------------ */
static void send_cb(void *request, ucs_status_t status, void *user_data) {
    (void)status; (void)user_data;
    g_send_outstanding--;
    ucp_request_free(request);
}

static void recv_cb(void *request, ucs_status_t status,
                    const ucp_tag_recv_info_t *info, void *user_data) {
    recv_ctx_t *r = (recv_ctx_t *)user_data;
    r->status    = status;
    r->length    = info->length;
    r->completed = 1;
    (void)request;
}

static void req_init(void *request) {
    memset(request, 0, REQ_SIZE);
}

/* ------------------------------------------------------------------ */
/* Schedule loading (from ucx_sender.c)                                */
/* ------------------------------------------------------------------ */
static int load_schedule_file(const char *path, schedule_t *sched) {
    FILE *f = fopen(path, "rb");
    if (!f) { perror("[node] fopen schedule"); return -1; }

    sched_header_t hdr;
    if (fread(&hdr, sizeof(hdr), 1, f) != 1) {
        fprintf(stderr, "[node] failed to read schedule header\n");
        fclose(f); return -1;
    }
    if (memcmp(hdr.magic, "SCHD", 4) != 0) {
        fprintf(stderr, "[node] invalid schedule magic\n");
        fclose(f); return -1;
    }
    if (hdr.version != SCHED_VERSION_V2) {
        fprintf(stderr, "[node] unsupported schedule version %u "
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
        fprintf(stderr, "[node] short read: %zu / %u doubles\n",
                rd, hdr.num_windows);
        free(sched->rates_bps); return -1;
    }

    double sum = 0, max_rate = 0;
    uint32_t active = 0;
    for (uint32_t i = 0; i < sched->num_windows; i++) {
        double r = sched->rates_bps[i];
        if (r < 0) sched->rates_bps[i] = 0;
        if (r > RATE_EPSILON) active++;
        sum += r;
        if (r > max_rate) max_rate = r;
    }
    double avg = sched->num_windows > 0 ? sum / sched->num_windows : 0;
    double dur = (double)sched->num_windows * sched->window_ns / 1e9;

    printf("[node/send] schedule: %u windows x %.2f ms = %.1f s, "
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

    printf("[node/send] constant rate: %.1f Gbps, %u windows x %.3f ms\n",
           rate_bps * 8.0 / 1e9, n, window_ns / 1e6);
}

/* ------------------------------------------------------------------ */
/* Sender EP error callback                                            */
/* ------------------------------------------------------------------ */
static void sender_ep_err_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
    (void)arg; (void)ep;
    fprintf(stderr, "[node/send] endpoint error: %s (will reconnect)\n",
            ucs_status_string(status));
    g_send_stop = 1;  /* Pause sender, main loop will reconnect */
    g_ep_error  = 1;
}

/* ------------------------------------------------------------------ */
/* Receiver EP error callback                                          */
/* ------------------------------------------------------------------ */
static void recv_ep_err_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
    (void)arg; (void)ep;
    if (g_receiving) {
        printf("[node/recv] peer disconnected: %s\n",
               ucs_status_string(status));
        g_recv_stop = 1;  /* Only stop receiver, not sender */
    }
}

/* ------------------------------------------------------------------ */
/* Receiver connection handler                                         */
/* ------------------------------------------------------------------ */
static void conn_request_cb(ucp_conn_request_h conn_request, void *arg) {
    ucp_worker_h worker = (ucp_worker_h)arg;

    /* If we already have a client EP (reconnection), force-close the old one */
    if (g_recv_client_ep != NULL) {
        printf("[node/recv] new client arriving, closing old EP...\n");
        ucp_request_param_t cp = {
            .op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS,
            .flags        = UCP_EP_CLOSE_FLAG_FORCE,
        };
        ucs_status_ptr_t cr = ucp_ep_close_nbx(g_recv_client_ep, &cp);
        if (cr != NULL && !UCS_PTR_IS_ERR(cr))
            ucp_request_free(cr);  /* best-effort, don't block in callback */
        g_recv_client_ep = NULL;
    }

    ucp_ep_params_t ep_params = {
        .field_mask   = UCP_EP_PARAM_FIELD_CONN_REQUEST |
                        UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                        UCP_EP_PARAM_FIELD_ERR_HANDLER,
        .conn_request = conn_request,
        .err_mode     = UCP_ERR_HANDLING_MODE_PEER,
        .err_handler  = { .cb = recv_ep_err_cb, .arg = NULL },
    };
    ucs_status_t st = ucp_ep_create(worker, &ep_params, &g_recv_client_ep);
    if (st != UCS_OK) {
        fprintf(stderr, "[node/recv] accept: %s\n", ucs_status_string(st));
    } else {
        printf("[node/recv] client connected\n");
        g_recv_new_client = 1;  /* Signal main loop to restart receiving */
    }
}

/* ------------------------------------------------------------------ */
/* Connect to send target (with retry/settle logic)                    */
/* ------------------------------------------------------------------ */
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
        fprintf(stderr, "[node/send] ucp_ep_create: %s\n",
                ucs_status_string(st));
        return NULL;
    }
    return ep;
}

/* ------------------------------------------------------------------ */
/* Post one non-blocking send                                          */
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
        g_send_outstanding++;

    *total_bytes += msg_size;
    *total_msgs  += 1;
    *stats_bytes += msg_size;
    return 0;
}

/* ------------------------------------------------------------------ */
/* Usage                                                               */
/* ------------------------------------------------------------------ */
static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s [options]\n\n"
        "Options:\n"
        "  --send-ip <addr>    Target IP to send to (omit for recv-only)\n"
        "  --send-port <port>  Target port to send to (default: 18515)\n"
        "  --recv-port <port>  Port to listen on (omit for send-only)\n"
        "  --schedule <file>   Binary V2 rate schedule for sending\n"
        "  --rate-bps <N>      Constant send rate in bytes/sec\n"
        "  --duration <sec>    Run duration (default: from schedule or 3600)\n"
        "  --msg-size <bytes>  Message size (default: %d)\n"
        "  --csv <path>        Write per-1ms recv BW to CSV file\n"
        "  --help              Show this help\n",
        prog, DEFAULT_MSG_SIZE);
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    const char *send_ip       = NULL;
    uint16_t    send_port     = 18515;
    int         recv_port_set = 0;
    uint16_t    recv_port     = 0;
    const char *schedule_path = NULL;
    double      const_rate    = -1;
    double      duration_sec  = 0;
    uint32_t    msg_size      = DEFAULT_MSG_SIZE;
    const char *csv_path      = NULL;

    static struct option long_opts[] = {
        {"send-ip",   required_argument, 0, 'I'},
        {"send-port", required_argument, 0, 'P'},
        {"recv-port", required_argument, 0, 'R'},
        {"schedule",  required_argument, 0, 's'},
        {"rate-bps",  required_argument, 0, 'r'},
        {"duration",  required_argument, 0, 'D'},
        {"msg-size",  required_argument, 0, 'm'},
        {"csv",       required_argument, 0, 'c'},
        {"help",      no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };

    int c;
    while ((c = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (c) {
            case 'I': send_ip = optarg; break;
            case 'P': send_port = (uint16_t)atoi(optarg); break;
            case 'R': recv_port = (uint16_t)atoi(optarg);
                      recv_port_set = 1; break;
            case 's': schedule_path = optarg; break;
            case 'r': const_rate = strtod(optarg, NULL); break;
            case 'D': duration_sec = strtod(optarg, NULL); break;
            case 'm': msg_size = (uint32_t)atol(optarg); break;
            case 'c': csv_path = optarg; break;
            case 'h': usage(argv[0]); return 0;
            default:  usage(argv[0]); return 1;
        }
    }

    int do_send = (send_ip != NULL);
    int do_recv = recv_port_set;

    if (!do_send && !do_recv) {
        fprintf(stderr, "[node] must specify --send-ip and/or --recv-port\n");
        usage(argv[0]);
        return 1;
    }

    printf("[node] mode: %s%s%s\n",
           do_send ? "sender" : "",
           (do_send && do_recv) ? " + " : "",
           do_recv ? "receiver" : "");

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    /* ---- Load or create schedule (sender only) ---- */
    schedule_t sched = { 0 };
    uint64_t   total_windows = 0;

    if (do_send) {
        if (schedule_path) {
            if (load_schedule_file(schedule_path, &sched) < 0)
                return 1;
            msg_size = sched.msg_size;
        } else {
            if (const_rate < 0) const_rate = 100e9;
            if (duration_sec <= 0) duration_sec = 3600;
            make_constant_rate_schedule(&sched, const_rate,
                                        DEFAULT_WINDOW_NS, msg_size,
                                        duration_sec);
        }

        double sched_dur = (double)sched.num_windows * sched.window_ns / 1e9;
        if (duration_sec <= 0) duration_sec = sched_dur;
        total_windows = (uint64_t)ceil(
            duration_sec * 1e9 / (double)sched.window_ns);

        printf("[node/send] duration=%.1f s (%llu windows)%s\n",
               duration_sec, (unsigned long long)total_windows,
               duration_sec > sched_dur ? " (loops)" : "");
    } else {
        /* Receiver-only: set duration */
        if (duration_sec <= 0) duration_sec = 0;  /* 0 = unlimited */
    }

    /* ---- UCX init (single context, single worker) ---- */
    ucp_params_t ucp_params = {
        .field_mask   = UCP_PARAM_FIELD_FEATURES |
                        UCP_PARAM_FIELD_REQUEST_SIZE |
                        UCP_PARAM_FIELD_REQUEST_INIT,
        .features     = UCP_FEATURE_TAG,
        .request_size = REQ_SIZE,
        .request_init = req_init,
    };
    ucp_context_h context;
    ucs_status_t st = ucp_init(&ucp_params, NULL, &context);
    if (st != UCS_OK) {
        fprintf(stderr, "[node] ucp_init: %s\n", ucs_status_string(st));
        return 1;
    }

    ucp_worker_params_t wparams = {
        .field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCS_THREAD_MODE_SINGLE,
    };
    ucp_worker_h worker;
    st = ucp_worker_create(context, &wparams, &worker);
    if (st != UCS_OK) {
        fprintf(stderr, "[node] ucp_worker_create: %s\n",
                ucs_status_string(st));
        ucp_cleanup(context);
        return 1;
    }

    /* ---- Receiver: create listener ---- */
    ucp_listener_h listener = NULL;
    if (do_recv) {
        struct sockaddr_in la = {
            .sin_family      = AF_INET,
            .sin_port        = htons(recv_port),
            .sin_addr.s_addr = INADDR_ANY,
        };
        ucp_listener_params_t lp = {
            .field_mask       = UCP_LISTENER_PARAM_FIELD_SOCK_ADDR |
                                UCP_LISTENER_PARAM_FIELD_CONN_HANDLER,
            .sockaddr.addr    = (struct sockaddr *)&la,
            .sockaddr.addrlen = sizeof(la),
            .conn_handler.cb  = conn_request_cb,
            .conn_handler.arg = worker,
        };
        st = ucp_listener_create(worker, &lp, &listener);
        if (st != UCS_OK) {
            fprintf(stderr, "[node/recv] ucp_listener_create: %s\n",
                    ucs_status_string(st));
            ucp_worker_destroy(worker);
            ucp_cleanup(context);
            return 1;
        }
        printf("[node/recv] listening on port %u\n", recv_port);
        if (csv_path)
            printf("[node/recv] CSV output: %s\n", csv_path);
    }

    /* ---- Sender: connect with retry ---- */
    ucp_ep_h send_ep = NULL;
    if (do_send) {
        for (int attempt = 0; attempt < CONNECT_MAX_RETRIES; attempt++) {
            if (g_stop) break;
            printf("[node/send] connecting to %s:%u (attempt %d)...\n",
                   send_ip, send_port, attempt + 1);
            g_ep_error = 0;
            send_ep = connect_to_server(worker, send_ip, send_port);
            if (!send_ep) {
                sleep(CONNECT_RETRY_SEC);
                continue;
            }

            /* Settle: progress UCX to flush stale errors */
            printf("[node/send] settling connection...\n");
            {
                uint64_t settle_end = now_ns() + 1000000000ULL;
                while (now_ns() < settle_end && !g_stop && !g_ep_error)
                    ucp_worker_progress(worker);
            }

            if (g_ep_error) {
                printf("[node/send] EP error during settle, closing "
                       "and retrying...\n");
                ucp_request_param_t cp = {
                    .op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS,
                    .flags        = UCP_EP_CLOSE_FLAG_FORCE,
                };
                ucs_status_ptr_t cr = ucp_ep_close_nbx(send_ep, &cp);
                if (cr != NULL && !UCS_PTR_IS_ERR(cr)) {
                    uint64_t t0 = now_ns();
                    while (ucp_request_check_status(cr) == UCS_INPROGRESS &&
                           (now_ns() - t0) < 3000000000ULL)
                        ucp_worker_progress(worker);
                    ucp_request_free(cr);
                }
                send_ep = NULL;
                sleep(CONNECT_RETRY_SEC);
                continue;
            }

            printf("[node/send] connected\n");
            break;
        }
        if (!send_ep && !g_stop) {
            fprintf(stderr, "[node/send] failed to connect after %d attempts\n",
                    CONNECT_MAX_RETRIES);
            if (!do_recv) goto cleanup_ucx;
            /* If also receiving, continue as receiver-only */
            do_send = 0;
            printf("[node] falling back to receiver-only mode\n");
        }
    }
    if (g_stop) goto cleanup_ucx;

    /* ---- Receiver: wait for incoming connection ---- */
    if (do_recv) {
        printf("[node/recv] waiting for connection...\n");
        while (!g_recv_client_ep && !g_stop)
            ucp_worker_progress(worker);
        if (g_stop) goto cleanup_eps;
    }

    /* ---- Allocate buffers ---- */
    /* Send buffer */
    void *send_buf = NULL;
    if (do_send) {
        send_buf = malloc(msg_size);
        if (!send_buf) {
            fprintf(stderr, "[node] malloc send_buf failed\n");
            goto cleanup_eps;
        }
        memset(send_buf, 0xAB, msg_size);
    }

    /* Receive slots */
    size_t max_recv_msg = DEFAULT_MAX_RECV_MSG;
    recv_slot_t *slots = NULL;
    int n_recv_slots = 0;

    if (do_recv) {
        n_recv_slots = MAX_RECV_OUTSTANDING;
        slots = calloc(n_recv_slots, sizeof(recv_slot_t));
        if (!slots) {
            fprintf(stderr, "[node] calloc recv slots failed\n");
            goto cleanup_eps;
        }
        for (int i = 0; i < n_recv_slots; i++) {
            slots[i].buf = malloc(max_recv_msg);
            if (!slots[i].buf) {
                fprintf(stderr, "[node] malloc recv buf[%d] failed\n", i);
                goto cleanup_eps;
            }
        }
    }

    /* ---- CSV file ---- */
    FILE *csv_fp = NULL;
    if (csv_path && do_recv) {
        csv_fp = fopen(csv_path, "w");
        if (!csv_fp)
            perror("[node/recv] fopen csv");
        else
            fprintf(csv_fp, "window_ms,bytes,msgs,bw_gbps\n");
    }

    /* ---- Post initial receives ---- */
    #define POST_RECV(i) do { \
        slots[i].ctx.completed = 0; \
        slots[i].ctx.length    = 0; \
        slots[i].ctx.status    = UCS_OK; \
        ucp_request_param_t _rp = { \
            .op_attr_mask = UCP_OP_ATTR_FIELD_CALLBACK | \
                            UCP_OP_ATTR_FIELD_USER_DATA, \
            .cb.recv      = recv_cb, \
            .user_data    = &slots[i].ctx, \
        }; \
        slots[i].req = ucp_tag_recv_nbx( \
            worker, slots[i].buf, max_recv_msg, TAG_DATA, TAG_MASK, &_rp); \
        slots[i].active = !UCS_PTR_IS_ERR(slots[i].req); \
    } while(0)

    if (do_recv) {
        for (int i = 0; i < n_recv_slots; i++)
            POST_RECV(i);
        printf("[node/recv] receiving (%d pipelined)...\n", n_recv_slots);
        g_receiving = 1;
    }

    /* ---- Initialize sender state ---- */
    token_bucket_t tb = {0};
    uint64_t send_total_bytes = 0, send_total_msgs = 0;
    uint64_t send_stats_bytes = 0, send_errors = 0;
    uint64_t send_window_idx  = 0;
    double   send_current_rate = 0;

    if (do_send) {
        tb_init(&tb, sched.rates_bps[0]);
        send_current_rate = sched.rates_bps[0];
        g_sending = 1;
        printf("[node/send] rate loop: msg_size=%u, window=%.2f ms, "
               "%llu windows\n",
               sched.msg_size, (double)sched.window_ns / 1e6,
               (unsigned long long)total_windows);
    }

    /* ---- Initialize receiver state ---- */
    uint64_t recv_window_start = 0, recv_window_end = 0;
    uint64_t recv_window_idx   = 0;
    uint64_t recv_win_bytes    = 0, recv_win_msgs  = 0;
    uint64_t recv_print_bytes  = 0, recv_print_msgs = 0;
    uint64_t recv_print_windows = 0;
    uint64_t recv_total_bytes  = 0, recv_total_msgs = 0;
    uint64_t recv_errors       = 0;

    /* ---- Main loop timing ---- */
    uint64_t loop_start = now_ns();

    if (do_recv) {
        recv_window_start = loop_start;
        recv_window_end   = loop_start + RECV_WINDOW_NS;
    }

    uint64_t send_last_stats = loop_start;
    uint64_t recv_start_ns   = loop_start;

    /* ================================================================ */
    /* Main event loop                                                   */
    /* ================================================================ */
    while (!g_stop) {
        /* Check overall duration */
        uint64_t now = now_ns();
        if (duration_sec > 0) {
            double elapsed = (double)(now - loop_start) / 1e9;
            if (elapsed >= duration_sec) break;
        }

        /* Handle sender-side errors: close dead EP and reconnect */
        if (g_send_stop && do_send && send_ep) {
            g_sending = 0;

            /* Drain outstanding sends */
            if (g_send_outstanding > 0) {
                printf("[node/send] draining %d outstanding sends...\n",
                       g_send_outstanding);
                uint64_t t0 = now_ns();
                while (g_send_outstanding > 0 &&
                       (now_ns() - t0) < 2000000000ULL)
                    ucp_worker_progress(worker);
            }

            /* Force-close the dead EP */
            ucp_request_param_t cp = {
                .op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS,
                .flags        = UCP_EP_CLOSE_FLAG_FORCE,
            };
            ucs_status_ptr_t cr = ucp_ep_close_nbx(send_ep, &cp);
            if (cr != NULL && !UCS_PTR_IS_ERR(cr)) {
                uint64_t t0 = now_ns();
                while (ucp_request_check_status(cr) == UCS_INPROGRESS &&
                       (now_ns() - t0) < 3000000000ULL)
                    ucp_worker_progress(worker);
                ucp_request_free(cr);
            }
            send_ep = NULL;

            /* Attempt reconnection */
            printf("[node/send] reconnecting to %s:%u...\n",
                   send_ip, send_port);
            sleep(CONNECT_RETRY_SEC);

            g_send_stop = 0;
            g_ep_error  = 0;
            for (int attempt = 0; attempt < 5 && !g_stop; attempt++) {
                g_ep_error = 0;
                send_ep = connect_to_server(worker, send_ip, send_port);
                if (!send_ep) {
                    sleep(CONNECT_RETRY_SEC);
                    continue;
                }
                /* Quick settle */
                uint64_t settle_end = now_ns() + 1000000000ULL;
                while (now_ns() < settle_end && !g_stop && !g_ep_error)
                    ucp_worker_progress(worker);
                if (g_ep_error) {
                    ucp_request_param_t cp2 = {
                        .op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS,
                        .flags        = UCP_EP_CLOSE_FLAG_FORCE,
                    };
                    ucs_status_ptr_t cr2 = ucp_ep_close_nbx(send_ep, &cp2);
                    if (cr2 != NULL && !UCS_PTR_IS_ERR(cr2)) {
                        uint64_t t0 = now_ns();
                        while (ucp_request_check_status(cr2) ==
                               UCS_INPROGRESS &&
                               (now_ns() - t0) < 3000000000ULL)
                            ucp_worker_progress(worker);
                        ucp_request_free(cr2);
                    }
                    send_ep = NULL;
                    sleep(CONNECT_RETRY_SEC);
                    continue;
                }
                printf("[node/send] reconnected\n");
                g_sending = 1;
                g_send_stop = 0;
                /* Reset token bucket for smooth restart */
                tb_init(&tb, sched.rates_bps[
                    send_window_idx % sched.num_windows]);
                break;
            }
            if (!send_ep) {
                printf("[node/send] reconnection failed, "
                       "continuing recv-only\n");
                do_send = 0;
            }
        }

        /* Handle receiver-side disconnect: pause recv, wait for reconnect */
        if (g_recv_stop && do_recv) {
            g_receiving = 0;

            /* Cancel outstanding receives */
            for (int i = 0; i < n_recv_slots; i++) {
                if (slots && slots[i].active && slots[i].req != NULL) {
                    ucp_request_cancel(worker, slots[i].req);
                }
            }
            /* Progress to flush cancellations */
            {
                uint64_t t0 = now_ns();
                while ((now_ns() - t0) < 200000000ULL)  /* 0.2s */
                    ucp_worker_progress(worker);
            }
            for (int i = 0; i < n_recv_slots; i++) {
                if (slots && slots[i].active && slots[i].req != NULL)
                    ucp_request_free(slots[i].req);
                slots[i].active = 0;
                slots[i].req = NULL;
            }

            printf("[node/recv] receiver paused, waiting for reconnect...\n");
            do_recv = 0;
            g_recv_stop = 0;
        }

        /* Handle receiver reconnection: new client connected */
        if (g_recv_new_client && !do_recv && slots) {
            g_recv_new_client = 0;
            g_recv_stop = 0;
            do_recv = 1;
            g_receiving = 1;

            /* Repost all receives on the new EP */
            for (int i = 0; i < n_recv_slots; i++)
                POST_RECV(i);

            printf("[node/recv] receiver restarted (new client)\n");
        }

        /* Check if sender has finished all windows */
        if (do_send && send_window_idx >= total_windows) {
            if (!do_recv) break;
            /* Sender done, but keep receiving */
            do_send = 0;
            g_sending = 0;
            printf("[node/send] all windows completed, continuing recv-only\n");
        }

        /* ---- UCX progress ---- */
        ucp_worker_progress(worker);

        now = now_ns();

        /* ============================================================ */
        /* Receiver: harvest completions + window tracking               */
        /* ============================================================ */
        if (do_recv) {
            /* Window boundary check */
            while (now >= recv_window_end) {
                double win_elapsed = (double)RECV_WINDOW_NS / 1e9;
                double bw_gbps = (recv_win_bytes * 8.0) / (win_elapsed * 1e9);

                if (csv_fp) {
                    double win_ms = (double)(recv_window_start - recv_start_ns)
                                    / 1e6;
                    fprintf(csv_fp, "%.3f,%" PRIu64 ",%" PRIu64 ",%.4f\n",
                            win_ms, recv_win_bytes, recv_win_msgs, bw_gbps);
                }

                recv_print_bytes += recv_win_bytes;
                recv_print_msgs  += recv_win_msgs;
                recv_print_windows++;

                if (recv_print_windows >= RECV_PRINT_EVERY_N) {
                    double prt_elapsed =
                        (double)(RECV_PRINT_EVERY_N * RECV_WINDOW_NS) / 1e9;
                    double prt_gbps =
                        (recv_print_bytes * 8.0) / (prt_elapsed * 1e9);
                    double t_sec = (double)(now - recv_start_ns) / 1e9;
                    printf("[node/recv] t=%.1fs: %.2f Gbps "
                           "(%" PRIu64 " msgs, %u-window avg)\n",
                           t_sec, prt_gbps, recv_print_msgs,
                           RECV_PRINT_EVERY_N);
                    recv_print_bytes   = 0;
                    recv_print_msgs    = 0;
                    recv_print_windows = 0;
                }

                recv_win_bytes    = 0;
                recv_win_msgs     = 0;
                recv_window_idx++;
                recv_window_start = recv_window_end;
                recv_window_end  += RECV_WINDOW_NS;
            }

            /* Harvest completed receives and repost */
            for (int i = 0; i < n_recv_slots; i++) {
                if (!slots[i].active) {
                    recv_errors++;
                    POST_RECV(i);
                    continue;
                }

                int done = 0;
                if (slots[i].req == NULL) {
                    done = 1;
                } else if (slots[i].ctx.completed) {
                    done = 1;
                }

                if (done) {
                    if (slots[i].ctx.status == UCS_OK) {
                        recv_win_bytes   += slots[i].ctx.length;
                        recv_win_msgs++;
                        recv_total_bytes += slots[i].ctx.length;
                        recv_total_msgs++;
                    } else {
                        recv_errors++;
                    }
                    if (slots[i].req != NULL)
                        ucp_request_free(slots[i].req);
                    POST_RECV(i);
                }
            }
        }

        /* ============================================================ */
        /* Sender: token-bucket paced, non-blocking                      */
        /* ============================================================ */
        if (do_send && send_ep) {
            /* Determine current window */
            uint64_t elapsed_ns = now - loop_start;
            uint64_t cur_window = elapsed_ns / sched.window_ns;

            if (cur_window >= total_windows) {
                /* Will be caught at top of loop */
                continue;
            }

            /* Update rate if window changed */
            if (cur_window != send_window_idx) {
                send_window_idx = cur_window;
                double new_rate =
                    sched.rates_bps[cur_window % sched.num_windows];
                tb.rate_bps = new_rate;
                send_current_rate = new_rate;
            }

            /* If rate is basically zero, skip */
            if (send_current_rate >= RATE_EPSILON) {
                /* Try to send if we have tokens and pipeline slots */
                if (g_send_outstanding < MAX_SEND_OUTSTANDING) {
                    if (tb_try_consume(&tb, sched.msg_size) == 0) {
                        if (post_send(send_ep, send_buf, sched.msg_size,
                                      &send_total_bytes, &send_total_msgs,
                                      &send_stats_bytes) < 0) {
                            send_errors++;
                            if (send_errors > 100) { g_stop = 1; }
                        }
                    }
                }
            }

            /* Sender stats every STATS_INTERVAL_SEC */
            double since_stats = (double)(now - send_last_stats) / 1e9;
            if (since_stats >= STATS_INTERVAL_SEC) {
                double bw_gbps =
                    (send_stats_bytes * 8.0) / (since_stats * 1e9);
                double elapsed_s = (double)(now - loop_start) / 1e9;
                double cur_rate_gbps = send_current_rate * 8.0 / 1e9;
                printf("[node/send] t=%.1fs: %.2f Gbps sent "
                       "(target=%.2f Gbps, total=%.2f GB, "
                       "%" PRIu64 " msgs)\n",
                       elapsed_s, bw_gbps, cur_rate_gbps,
                       send_total_bytes / 1e9, send_total_msgs);
                send_stats_bytes = 0;
                send_last_stats  = now;
            }
        }
    }

    #undef POST_RECV

    /* ================================================================ */
    /* Drain & cleanup                                                   */
    /* ================================================================ */

    g_sending   = 0;
    g_receiving = 0;

    /* Drain outstanding sends */
    if (send_ep && g_send_outstanding > 0) {
        printf("[node/send] draining %d outstanding sends...\n",
               g_send_outstanding);
        uint64_t t0 = now_ns();
        while (g_send_outstanding > 0 && (now_ns() - t0) < 5000000000ULL)
            ucp_worker_progress(worker);
    }

    /* Cancel outstanding receives */
    if (slots) {
        for (int i = 0; i < n_recv_slots; i++) {
            if (slots[i].active && slots[i].req != NULL)
                ucp_request_cancel(worker, slots[i].req);
        }
        {
            uint64_t t0 = now_ns();
            while ((now_ns() - t0) < 500000000ULL)
                ucp_worker_progress(worker);
        }
        for (int i = 0; i < n_recv_slots; i++) {
            if (slots[i].active && slots[i].req != NULL)
                ucp_request_free(slots[i].req);
        }
    }

    /* Flush last partial recv window to CSV */
    if (csv_fp && recv_win_bytes > 0) {
        double win_ms = (double)(recv_window_start - recv_start_ns) / 1e6;
        uint64_t actual_ns = now_ns() - recv_window_start;
        double bw_gbps = actual_ns > 0 ?
            (recv_win_bytes * 8.0) / ((double)actual_ns / 1e9 * 1e9) : 0;
        fprintf(csv_fp, "%.3f,%" PRIu64 ",%" PRIu64 ",%.4f\n",
                win_ms, recv_win_bytes, recv_win_msgs, bw_gbps);
    }
    if (csv_fp) {
        fclose(csv_fp);
        printf("[node/recv] CSV written: %s (%" PRIu64 " windows)\n",
               csv_path, recv_window_idx + 1);
    }

    /* ---- Sender summary ---- */
    if (send_total_msgs > 0 || send_ep) {
        double send_sec = (double)(now_ns() - loop_start) / 1e9;
        printf("\n=== Sender Summary ===\n");
        printf("  Duration      : %.1f sec\n", send_sec);
        printf("  Total sent    : %.2f GB (%" PRIu64 " msgs)\n",
               send_total_bytes / 1e9, send_total_msgs);
        if (send_sec > 0)
            printf("  Avg BW        : %.2f Gbps\n",
                   send_total_bytes * 8.0 / (send_sec * 1e9));
        printf("  Msg size      : %u bytes\n", sched.msg_size);
        if (send_errors > 0)
            printf("  Send errors   : %" PRIu64 "\n", send_errors);
    }

    /* ---- Receiver summary ---- */
    if (recv_total_msgs > 0 || g_recv_client_ep) {
        double recv_sec = (double)(now_ns() - recv_start_ns) / 1e9;
        printf("\n=== Receiver Summary ===\n");
        printf("  Duration   : %.1f sec\n", recv_sec);
        printf("  Total recv : %.2f GB (%" PRIu64 " msgs)\n",
               recv_total_bytes / 1e9, recv_total_msgs);
        if (recv_sec > 0)
            printf("  Avg BW     : %.2f Gbps\n",
                   recv_total_bytes * 8.0 / (recv_sec * 1e9));
        printf("  Windows    : %" PRIu64 " (1 ms each)\n",
               recv_window_idx + 1);
        if (recv_errors > 0)
            printf("  Errors     : %" PRIu64 "\n", recv_errors);
    }

    /* ---- Free buffers ---- */
    free(send_buf);
    if (slots) {
        for (int i = 0; i < n_recv_slots; i++)
            free(slots[i].buf);
        free(slots);
    }

cleanup_eps:
    /* Close send EP */
    if (send_ep) {
        ucp_request_param_t cp = {
            .op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS,
            .flags        = UCP_EP_CLOSE_FLAG_FORCE,
        };
        ucs_status_ptr_t cr = ucp_ep_close_nbx(send_ep, &cp);
        if (cr != NULL && !UCS_PTR_IS_ERR(cr)) {
            uint64_t t0 = now_ns();
            while (ucp_request_check_status(cr) == UCS_INPROGRESS &&
                   (now_ns() - t0) < 3000000000ULL)
                ucp_worker_progress(worker);
            ucp_request_free(cr);
        }
    }
    /* Close recv client EP */
    if (g_recv_client_ep) {
        ucp_request_param_t cp = {
            .op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS,
            .flags        = UCP_EP_CLOSE_FLAG_FORCE,
        };
        ucs_status_ptr_t cr = ucp_ep_close_nbx(g_recv_client_ep, &cp);
        if (cr != NULL && !UCS_PTR_IS_ERR(cr)) {
            uint64_t t0 = now_ns();
            while (ucp_request_check_status(cr) == UCS_INPROGRESS &&
                   (now_ns() - t0) < 3000000000ULL)
                ucp_worker_progress(worker);
            ucp_request_free(cr);
        }
    }

cleanup_ucx:
    if (listener) ucp_listener_destroy(listener);
    ucp_worker_destroy(worker);
    ucp_cleanup(context);
    free(sched.rates_bps);
    return 0;
}

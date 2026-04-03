/**
 * ucx_receiver.c — UCX traffic receiver with sub-ms BW tracking (RoCE)
 *
 * Listens on a port, accepts one UCX connection, and receives tag
 * messages.  Tracks received bytes per 1ms window and optionally
 * writes per-window BW to a CSV file.
 *
 * Build:
 *   gcc -O3 -march=native -o ucx_receiver ucx_receiver.c \
 *       $(pkg-config --cflags --libs ucx)
 *
 * Usage:
 *   UCX_TLS=rc UCX_NET_DEVICES=mlx5_1:1 \
 *       ./ucx_receiver 18515 --duration 3600 --csv recv_bw.csv
 */

#define _GNU_SOURCE
#include <ucp/api/ucp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <stdint.h>
#include <inttypes.h>
#include <time.h>
#include <signal.h>
#include <netinet/in.h>
#include <arpa/inet.h>
#include <unistd.h>
#include <getopt.h>

/* ------------------------------------------------------------------ */
/* Tunables                                                            */
/* ------------------------------------------------------------------ */
#define TAG_DATA            0xBEEFCAFEULL
#define TAG_MASK            UINT64_MAX
#define DEFAULT_MAX_MSG     (16 * 1024 * 1024)   /* 16 MB                */
#define WINDOW_NS           1000000ULL           /* 1 ms tracking window  */
#define PRINT_EVERY_N       5000                 /* print every N windows */
#define MAX_RECV_OUTSTANDING 256                 /* pipelined receives     */

static volatile int g_stop     = 0;
static volatile int g_receiving = 0;  /* set after first msg received */
static void sig_handler(int s) { (void)s; g_stop = 1; }

/* ------------------------------------------------------------------ */
/* Receive tracking                                                    */
/* ------------------------------------------------------------------ */
typedef struct {
    volatile int completed;
    ucs_status_t status;
    size_t       length;
} recv_ctx_t;

static void recv_cb(void *request, ucs_status_t status,
                    const ucp_tag_recv_info_t *info, void *user_data) {
    recv_ctx_t *r = (recv_ctx_t *)user_data;
    r->status    = status;
    r->length    = info->length;
    r->completed = 1;
    (void)request;
}

static void req_init(void *request) {
    memset(request, 0, sizeof(recv_ctx_t));
}

static inline uint64_t now_ns(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (uint64_t)ts.tv_sec * 1000000000ULL + ts.tv_nsec;
}

/* ------------------------------------------------------------------ */
/* Connection handler                                                  */
/* ------------------------------------------------------------------ */
static ucp_ep_h g_client_ep = NULL;

static void ep_err_cb(void *arg, ucp_ep_h ep, ucs_status_t status) {
    (void)arg; (void)ep;
    if (g_receiving) {
        printf("[receiver] peer disconnected: %s\n", ucs_status_string(status));
        g_stop = 1;
    }
    /* Ignore errors before receive loop is active (stale RDMA cleanup) */
}

static void conn_request_cb(ucp_conn_request_h conn_request, void *arg) {
    ucp_worker_h worker = (ucp_worker_h)arg;
    ucp_ep_params_t ep_params = {
        .field_mask   = UCP_EP_PARAM_FIELD_CONN_REQUEST |
                        UCP_EP_PARAM_FIELD_ERR_HANDLING_MODE |
                        UCP_EP_PARAM_FIELD_ERR_HANDLER,
        .conn_request = conn_request,
        .err_mode     = UCP_ERR_HANDLING_MODE_PEER,
        .err_handler  = { .cb = ep_err_cb, .arg = NULL },
    };
    ucs_status_t st = ucp_ep_create(worker, &ep_params, &g_client_ep);
    if (st != UCS_OK)
        fprintf(stderr, "[receiver] accept: %s\n", ucs_status_string(st));
    else
        printf("[receiver] client connected\n");
}

/* ------------------------------------------------------------------ */
/* Usage                                                               */
/* ------------------------------------------------------------------ */
static void usage(const char *prog) {
    fprintf(stderr,
        "Usage: %s <port> [options]\n\n"
        "Options:\n"
        "  --max-msg <bytes>   Max message size (default: %d)\n"
        "  --duration <sec>    Run duration, 0=unlimited (default: 0)\n"
        "  --csv <path>        Write per-1ms BW to CSV file\n"
        "\n"
        "Environment:\n"
        "  UCX_TLS=rc  UCX_NET_DEVICES=mlx5_1:1\n",
        prog, DEFAULT_MAX_MSG);
}

/* ------------------------------------------------------------------ */
/* Main                                                                */
/* ------------------------------------------------------------------ */
int main(int argc, char **argv) {
    if (argc < 2) { usage(argv[0]); return 1; }

    uint16_t    port         = (uint16_t)atoi(argv[1]);
    size_t      max_msg      = DEFAULT_MAX_MSG;
    double      duration_sec = 0;
    const char *csv_path     = NULL;

    static struct option long_opts[] = {
        {"max-msg",  required_argument, 0, 'm'},
        {"duration", required_argument, 0, 'D'},
        {"csv",      required_argument, 0, 'c'},
        {"help",     no_argument,       0, 'h'},
        {0, 0, 0, 0}
    };
    optind = 2;
    int c;
    while ((c = getopt_long(argc, argv, "", long_opts, NULL)) != -1) {
        switch (c) {
            case 'm': max_msg = (size_t)atol(optarg); break;
            case 'D': duration_sec = strtod(optarg, NULL); break;
            case 'c': csv_path = optarg; break;
            case 'h': usage(argv[0]); return 0;
            default:  usage(argv[0]); return 1;
        }
    }

    signal(SIGINT, sig_handler);
    signal(SIGTERM, sig_handler);

    /* ---- UCX init ---- */
    ucp_params_t ucp_params = {
        .field_mask   = UCP_PARAM_FIELD_FEATURES |
                        UCP_PARAM_FIELD_REQUEST_SIZE |
                        UCP_PARAM_FIELD_REQUEST_INIT,
        .features     = UCP_FEATURE_TAG,
        .request_size = sizeof(recv_ctx_t),
        .request_init = req_init,
    };
    ucp_context_h ctx;
    ucs_status_t st = ucp_init(&ucp_params, NULL, &ctx);
    if (st != UCS_OK) {
        fprintf(stderr, "[receiver] ucp_init: %s\n",
                ucs_status_string(st));
        return 1;
    }

    ucp_worker_params_t wparams = {
        .field_mask  = UCP_WORKER_PARAM_FIELD_THREAD_MODE,
        .thread_mode = UCS_THREAD_MODE_SINGLE,
    };
    ucp_worker_h worker;
    st = ucp_worker_create(ctx, &wparams, &worker);
    if (st != UCS_OK) {
        fprintf(stderr, "[receiver] ucp_worker_create: %s\n",
                ucs_status_string(st));
        return 1;
    }

    /* ---- Listener ---- */
    struct sockaddr_in la = {
        .sin_family      = AF_INET,
        .sin_port        = htons(port),
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
    ucp_listener_h listener;
    st = ucp_listener_create(worker, &lp, &listener);
    if (st != UCS_OK) {
        fprintf(stderr, "[receiver] ucp_listener_create: %s\n",
                ucs_status_string(st));
        return 1;
    }

    printf("[receiver] listening on port %u, max_msg=%zu\n", port, max_msg);
    if (csv_path)
        printf("[receiver] CSV output: %s\n", csv_path);
    printf("[receiver] waiting for connection...\n");

    while (!g_client_ep && !g_stop)
        ucp_worker_progress(worker);
    if (g_stop) goto cleanup;

        /* ---- Receive buffers (pipelined) ---- */
    typedef struct {
        void            *buf;
        recv_ctx_t       ctx;
        ucs_status_ptr_t req;
        int              active;   /* 1 = receive posted */
    } recv_slot_t;

    recv_slot_t *slots = calloc(MAX_RECV_OUTSTANDING, sizeof(recv_slot_t));
    if (!slots) { fprintf(stderr, "[receiver] calloc slots failed\n"); goto cleanup; }
    for (int i = 0; i < MAX_RECV_OUTSTANDING; i++) {
        slots[i].buf = malloc(max_msg);
        if (!slots[i].buf) {
            fprintf(stderr, "[receiver] malloc buf[%d] failed\n", i);
            goto cleanup;
        }
    }

    /* ---- CSV file ---- */
    FILE *csv_fp = NULL;
    if (csv_path) {
        csv_fp = fopen(csv_path, "w");
        if (!csv_fp) {
            perror("[receiver] fopen csv");
        } else {
            fprintf(csv_fp, "window_ms,bytes,msgs,bw_gbps\n");
        }
    }

    /* Helper: post a receive on slot i */
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
            worker, slots[i].buf, max_msg, TAG_DATA, TAG_MASK, &_rp); \
        slots[i].active = !UCS_PTR_IS_ERR(slots[i].req); \
    } while(0)

    /* Post all initial receives */
    for (int i = 0; i < MAX_RECV_OUTSTANDING; i++)
        POST_RECV(i);

    /* ---- Receive loop with 1ms window tracking ---- */
    uint64_t start_ns       = now_ns();
    uint64_t window_start   = start_ns;
    uint64_t window_end     = start_ns + WINDOW_NS;
    uint64_t window_idx     = 0;

    uint64_t win_bytes      = 0;
    uint64_t win_msgs       = 0;

    /* Aggregate stats for console printing */
    uint64_t print_bytes    = 0;
    uint64_t print_msgs     = 0;
    uint64_t print_windows  = 0;

    uint64_t total_bytes    = 0;
    uint64_t total_msgs     = 0;
    uint64_t errors         = 0;

    printf("[receiver] receiving (%d pipelined)...\n", MAX_RECV_OUTSTANDING);
    g_receiving = 1;

    while (!g_stop) {
        /* Check duration */
        if (duration_sec > 0) {
            double elapsed = (double)(now_ns() - start_ns) / 1e9;
            if (elapsed >= duration_sec) break;
        }

        /* Progress UCX */
        ucp_worker_progress(worker);

        /* Window boundary check */
        uint64_t now = now_ns();
        while (now >= window_end) {
            double win_elapsed = (double)WINDOW_NS / 1e9;
            double bw_gbps = (win_bytes * 8.0) / (win_elapsed * 1e9);

            if (csv_fp) {
                double win_ms = (double)(window_start - start_ns) / 1e6;
                fprintf(csv_fp, "%.3f,%" PRIu64 ",%" PRIu64 ",%.4f\n",
                        win_ms, win_bytes, win_msgs, bw_gbps);
            }

            print_bytes += win_bytes;
            print_msgs  += win_msgs;
            print_windows++;

            if (print_windows >= PRINT_EVERY_N) {
                double prt_elapsed = (double)(PRINT_EVERY_N * WINDOW_NS) / 1e9;
                double prt_gbps = (print_bytes * 8.0) / (prt_elapsed * 1e9);
                double t_sec = (double)(now - start_ns) / 1e9;
                printf("[receiver] t=%.1fs: %.2f Gbps "
                       "(%" PRIu64 " msgs, %u-window avg)\n",
                       t_sec, prt_gbps, print_msgs, PRINT_EVERY_N);
                print_bytes   = 0;
                print_msgs    = 0;
                print_windows = 0;
            }

            win_bytes    = 0;
            win_msgs     = 0;
            window_idx++;
            window_start = window_end;
            window_end  += WINDOW_NS;
        }

        /* Harvest completed receives and repost */
        for (int i = 0; i < MAX_RECV_OUTSTANDING; i++) {
            if (!slots[i].active) {
                errors++;
                POST_RECV(i);  /* retry */
                continue;
            }

            /* Check if completed (inline completion or callback) */
            int done = 0;
            if (slots[i].req == NULL) {
                /* Completed inline — but we need the callback to fire
                 * for length info, which it already did */
                done = 1;
            } else if (slots[i].ctx.completed) {
                done = 1;
            }

            if (done) {
                if (slots[i].ctx.status == UCS_OK) {
                    win_bytes   += slots[i].ctx.length;
                    win_msgs++;
                    total_bytes += slots[i].ctx.length;
                    total_msgs++;
                } else {
                    errors++;
                }
                /* Free request if non-NULL */
                if (slots[i].req != NULL)
                    ucp_request_free(slots[i].req);
                /* Repost */
                POST_RECV(i);
            }
        }
    }

    #undef POST_RECV

    /* Cancel any outstanding receives */
    for (int i = 0; i < MAX_RECV_OUTSTANDING; i++) {
        if (slots[i].active && slots[i].req != NULL) {
            ucp_request_cancel(worker, slots[i].req);
            /* Progress to let cancellation complete */
        }
    }
    /* Progress to flush cancellations */
    {
        uint64_t t0 = now_ns();
        while ((now_ns() - t0) < 500000000ULL)  /* 0.5s */
            ucp_worker_progress(worker);
    }
    for (int i = 0; i < MAX_RECV_OUTSTANDING; i++) {
        if (slots[i].active && slots[i].req != NULL)
            ucp_request_free(slots[i].req);
    }

    /* Flush last partial window to CSV */
    if (csv_fp && win_bytes > 0) {
        double win_ms = (double)(window_start - start_ns) / 1e6;
        uint64_t actual_ns = now_ns() - window_start;
        double bw_gbps = actual_ns > 0 ?
            (win_bytes * 8.0) / ((double)actual_ns / 1e9 * 1e9) : 0;
        fprintf(csv_fp, "%.3f,%" PRIu64 ",%" PRIu64 ",%.4f\n",
                win_ms, win_bytes, win_msgs, bw_gbps);
    }

    if (csv_fp) {
        fclose(csv_fp);
        printf("[receiver] CSV written: %s (%" PRIu64 " windows)\n",
               csv_path, window_idx + 1);
    }

    /* ---- Summary ---- */
    double total_s = (double)(now_ns() - start_ns) / 1e9;
    printf("\n=== Receiver Summary ===\n");
    printf("  Duration   : %.1f sec\n", total_s);
    printf("  Total recv : %.2f GB (%" PRIu64 " msgs)\n",
           total_bytes / 1e9, total_msgs);
    printf("  Avg BW     : %.2f Gbps\n",
           total_bytes * 8.0 / (total_s * 1e9));
    printf("  Windows    : %" PRIu64 " (1 ms each)\n", window_idx + 1);
    if (errors > 0)
        printf("  Errors     : %" PRIu64 "\n", errors);

    for (int i = 0; i < MAX_RECV_OUTSTANDING; i++)
        free(slots[i].buf);
    free(slots);

cleanup:
    ucp_listener_destroy(listener);
    if (g_client_ep) {
        ucp_request_param_t close_params = {
            .op_attr_mask = UCP_OP_ATTR_FIELD_FLAGS,
            .flags        = UCP_EP_CLOSE_FLAG_FORCE,
        };
        ucs_status_ptr_t close_req = ucp_ep_close_nbx(g_client_ep, &close_params);
        if (close_req != NULL && !UCS_PTR_IS_ERR(close_req)) {
            uint64_t t0 = now_ns();
            while (ucp_request_check_status(close_req) == UCS_INPROGRESS &&
                   (now_ns() - t0) < 3000000000ULL)
                ucp_worker_progress(worker);
            ucp_request_free(close_req);
        }
    }
    ucp_worker_destroy(worker);
    ucp_cleanup(ctx);
    return 0;
}

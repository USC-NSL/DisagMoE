#!/usr/bin/env bash
set -euo pipefail

# Build the UCX interference sender and receiver.
#
# Prerequisites:
#   - UCX (libucp) installed with pkg-config support
#   - gcc
#
# Usage:
#   ./build_ucx.sh          # build in this directory
#   ./build_ucx.sh clean    # remove binaries

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

if [ "${1:-}" = "clean" ]; then
    rm -f "$SCRIPT_DIR/ucx_sender" "$SCRIPT_DIR/ucx_receiver" "$SCRIPT_DIR/ucx_ring_node"
    echo "Cleaned."
    exit 0
fi

# Check for UCX
if ! pkg-config --exists ucx 2>/dev/null; then
    echo "ERROR: UCX not found via pkg-config."
    echo ""
    echo "If UCX is installed but pkg-config can't find it, try:"
    echo "  export PKG_CONFIG_PATH=/path/to/ucx/lib/pkgconfig:\$PKG_CONFIG_PATH"
    echo ""
    echo "To install UCX from source:"
    echo "  git clone https://github.com/openucx/ucx.git && cd ucx"
    echo "  ./autogen.sh && ./configure --prefix=/usr/local && make -j && sudo make install"
    exit 1
fi

UCX_CFLAGS="$(pkg-config --cflags ucx)"
UCX_LDFLAGS="$(pkg-config --libs ucx)"
CFLAGS="-O3 -march=native -Wall -Wextra -Wno-unused-parameter"

echo "Building UCX interference tools..."
echo "  UCX CFLAGS:  $UCX_CFLAGS"
echo "  UCX LDFLAGS: $UCX_LDFLAGS"

echo "  Compiling ucx_sender..."
gcc $CFLAGS $UCX_CFLAGS \
    -o "$SCRIPT_DIR/ucx_sender" "$SCRIPT_DIR/ucx_sender.c" \
    $UCX_LDFLAGS -lm

echo "  Compiling ucx_receiver..."
gcc $CFLAGS $UCX_CFLAGS \
    -o "$SCRIPT_DIR/ucx_receiver" "$SCRIPT_DIR/ucx_receiver.c" \
    $UCX_LDFLAGS -lm

echo "  Compiling ucx_ring_node..."
gcc $CFLAGS $UCX_CFLAGS \
    -o "$SCRIPT_DIR/ucx_ring_node" "$SCRIPT_DIR/ucx_ring_node.c" \
    $UCX_LDFLAGS -lm

echo ""
echo "Done. Binaries:"
echo "  $SCRIPT_DIR/ucx_sender"
echo "  $SCRIPT_DIR/ucx_receiver"
echo "  $SCRIPT_DIR/ucx_ring_node"
echo ""
echo "Test:"
echo "  UCX_TLS=rc UCX_NET_DEVICES=mlx5_1:1 ./ucx_receiver 18515"
echo "  UCX_TLS=rc UCX_NET_DEVICES=mlx5_1:1 ./ucx_sender <peer_ip> 18515 --rate-bps 1000000000 --duration 10"

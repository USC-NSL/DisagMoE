# Clock Synchronization (PTP) for Sphere-16

Uses **linuxptp** with hardware timestamping on the Mellanox NIC (`ens1f1np1`, `/dev/ptp2`).
sgpu0 is the PTP grandmaster; all workers are slaves.

Achieves **<50 ns** inter-node HW clock sync and **<1 μs** system clock sync.

## Prerequisites

`linuxptp` is already installed (`ptp4l`, `phc2sys`, `phc_ctl`).
Verify hardware timestamping is available:

```bash
ethtool -T ens1f1np1 | grep hardware
```

## Step 1: Start PTP master on sgpu0

```bash
sudo nohup ptp4l -m -2 -i ens1f1np1 --priority1 128 --priority2 128 > /tmp/ptp4l.log 2>&1 &
```

## Step 2: Start PTP slaves on all workers

```bash
for node in sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  ssh $node "sudo nohup ptp4l -m -2 -i ens1f1np1 -s > /tmp/ptp4l.log 2>&1 &"
done
```

Wait ~5 seconds, then verify all workers show `SLAVE` state with sub-μs offsets:

```bash
for node in sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  echo -n "$node: "; ssh $node "tail -1 /tmp/ptp4l.log"
done
```

## Step 3: Step-set system clocks from PTP HW clocks

The system clocks may be tens of seconds off from the HW clocks.
Force-set them first so `phc2sys` doesn't have to slew for ages:

```bash
for node in sgpu0 sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  ssh $node 'PTP_TIME=$(sudo phc_ctl ens1f1np1 get 2>&1 | grep -oP "[\d.]+(?= or)"); sudo date -s @$PTP_TIME'
done
```

## Step 4: Start phc2sys to discipline system clocks

```bash
for node in sgpu0 sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  ssh $node "sudo nohup phc2sys -s ens1f1np1 -c CLOCK_REALTIME -w -m -O 0 > /tmp/phc2sys.log 2>&1 &"
done
```

Converges to <1 μs within a few minutes. Check with:

```bash
for node in sgpu0 sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  echo -n "$node: "; ssh $node "tail -1 /tmp/phc2sys.log"
done
```

## Teardown

```bash
for node in sgpu0 sgpu2 sgpu3 sgpu4 sgpu6 sgpu7 sgpu8 sgpu9; do
  ssh $node "sudo killall ptp4l phc2sys 2>/dev/null"
done
```

## Notes

- `-2` = IEEE 802.3 (Layer 2) transport — lower latency than UDP for same-rack RoCE.
- `-s` = slave-only mode on workers.
- `-w` = wait for `ptp4l` to sync before adjusting (phc2sys talks to ptp4l via UDS).
- `-O 0` = no UTC-TAI offset (both clocks use the same epoch).
- Processes survive SSH disconnect (`nohup`) but not reboot.

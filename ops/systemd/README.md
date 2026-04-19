# ops/systemd

Systemd units for the training host.

## `nvidia-pl.service`

Caps RTX 5090 sustained power draw at 500 W to protect the PSU against
transient spikes (see `docs/setup.pitfalls.md` → RTX 5090 Power Cap).

Install:

```bash
sudo cp ops/systemd/nvidia-pl.service /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now nvidia-pl.service
systemctl status nvidia-pl.service  # expect: active (exited)
```

Adjust the `-pl 500` value in the unit file if the hardware or PSU changes.

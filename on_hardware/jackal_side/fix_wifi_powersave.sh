#!/usr/bin/env bash
# Robot side: turn Wi-Fi power saving off now and every time wlp3s0 (re)connects.
# Power saving on caused ~80% packet loss / 300-600 ms ping to the OneMap server.
#
#   sudo bash ~/onemap_jackal/fix_wifi_powersave.sh
set -e
IFACE=wlp3s0
HOOK=/etc/networkd-dispatcher/routable.d/50-wifi-powersave-off
OLD_RULE=/etc/udev/rules.d/70-wifi-powersave-off.rules   # ran too early at boot, didn't stick

[ "$(id -u)" -eq 0 ] || { echo "run with sudo: sudo bash $0"; exit 1; }

/usr/sbin/iw dev "$IFACE" set power_save off

cat > "$HOOK" <<EOF
#!/bin/sh
[ "\$IFACE" = "$IFACE" ] && /usr/sbin/iw dev $IFACE set power_save off
exit 0
EOF
chmod 755 "$HOOK"
echo "installed $HOOK"

if [ -e "$OLD_RULE" ]; then rm "$OLD_RULE"; echo "removed $OLD_RULE"; fi

/usr/sbin/iw dev "$IFACE" get power_save

#!/bin/sh
# snapraid-hook.sh -- hold the WhisperJAV fleet while SnapRAID works the array.
#
# Install as snapraid-daemon's `hook_script` (snapraidd.conf):
#
#     hook_script = /usr/local/bin/snapraid-hook.sh
#
# and, optionally, so the hold starts BEFORE the disks spin up rather than
# once sync begins:
#
#     notify_start = /usr/local/bin/snapraid-hook.sh maintenance-start
#
# snapraidd runs the hook with one argument -- task-begin, task-end or
# task-error -- around every array-touching task (sync, scrub, fix, check,
# diff).  Daemons built from 2026-06-08 on (release v1.12+) also pass the
# details in SNAPRAID_COMMAND / SNAPRAID_HIGH_COMMAND; older ones (1.11.x)
# pass nothing but the event, so with those every hooked task arms the hold
# and HOLD_COMMANDS is moot.  The hook does NOT fire around spin-up (up),
# spin-down (down), probe or report, so a hook alone cannot bracket the whole
# maintenance sequence.  This script therefore never *releases* the hold; it
# only sets and extends it:
#
#   maintenance-start / task-begin   hold for HOLD_TTL_SECONDS from now
#   task-end / task-error            extend the hold to now + HOLD_TAIL_SECONDS,
#                                    covering the probe / spin-down / report
#                                    that follow the last hooked task
#
# Every call is "extend only": the deadline can move later, never earlier, so
# this can never cut short a hold someone set by hand or the hub's own daily
# maintenance window.  The hub's window is the floor (it covers the fixed
# 02:55-04:00 slot whether or not anything fires); this script is what makes
# the hold grow with the array when a sync runs long.  The fleet resumes
# when the later of the two deadlines passes -- nothing has to call back.
#
# Contract with snapraidd, and why the script is shaped the way it is:
#   * A non-zero exit from task-begin ABORTS the task.  This script always
#     exits 0: a hub that is down must never cancel the night's sync.
#   * The daemon kills the hook after 300 s.  Nothing here waits on anything
#     but a short curl.
#   * The environment is scrubbed before the hook runs, so settings live in
#     this file (or in HOOK_CONF below), not in the daemon's environment.
#   * The daemon's own rules: absolute path, owned by root or the daemon
#     user, not group- or world-writable, executable, no setuid, one hard
#     link.  `install -o root -g root -m 0755` satisfies them.

HUB_URL="http://192.168.11.50:8787"
HUB_USER=""                 # only if the hub sets HUB_USER / HUB_PASS
HUB_PASS=""
HOLD_COMMANDS="sync scrub"  # space-separated; the hooked tasks that (re)arm the hold
                            # (only honoured when the daemon passes SNAPRAID_COMMAND)
HOLD_TTL_SECONDS=7200       # 2 h from a task's start; the fail-open ceiling, not the expected duration
HOLD_TAIL_SECONDS=900       # 15 min after a hooked task ends, for probe + spin-down + report
HOLD_ABORT_IN_FLIGHT=1      # 1: abort (requeue) files being transcribed; 0: let them finish
HOOK_CONF="/usr/local/etc/snapraid-hook.conf"   # optional: overrides the values above

# shellcheck disable=SC1090
[ -r "$HOOK_CONF" ] && . "$HOOK_CONF"

event="${1:-$SNAPRAID_EVENT}"
cmd="${SNAPRAID_COMMAND:-}"
high="${SNAPRAID_HIGH_COMMAND:-}"
me="snapraid-hook"

say() { logger -t "$me" "$*" 2>/dev/null || echo "$me: $*" >&2; }

# notify_start pipes a message into stdin; drain it so the daemon never sees
# a broken pipe.  (Harmless for hook invocations, which get no stdin.)
[ "$event" = "maintenance-start" ] && cat >/dev/null 2>&1

# Filter by command only when the daemon tells us which one it is.  A 1.11.x
# daemon sets no SNAPRAID_* variables at all; then every hooked task counts,
# which is the safe direction (an extra hold on a diff costs nothing).
if [ "$event" != "maintenance-start" ] && [ -n "$cmd" ]; then
    wanted=0
    for c in $HOLD_COMMANDS; do
        [ "$c" = "$cmd" ] && wanted=1
    done
    [ "$wanted" -eq 1 ] || exit 0      # e.g. the diff before sync
fi
what="${cmd:-task}"                    # for messages when the daemon named no command

auth=""
[ -n "$HUB_USER" ] && auth="-u $HUB_USER:$HUB_PASS"
abort=false
[ "$HOLD_ABORT_IN_FLIGHT" = "1" ] && abort=true

hold() {
    # $1 = reason, $2 = ttl seconds.  Extend-only; short timeouts; never fatal.
    # shellcheck disable=SC2086
    curl -fsS $auth --max-time 10 --retry 2 --retry-delay 2 \
         -H "Content-Type: application/json" -X POST "$HUB_URL/api/hold" \
         --data "{\"held\": true, \"reason\": \"$1\", \"ttl_seconds\": $2, \"extend_only\": true, \"abort_in_flight\": $abort, \"set_by\": \"$me@$(hostname)\"}" \
         >/dev/null 2>&1
}

case "$event" in
    maintenance-start)
        if hold "snapraid maintenance starting" "$HOLD_TTL_SECONDS"; then
            say "fleet held ahead of maintenance (ttl ${HOLD_TTL_SECONDS}s)"
        else
            say "WARNING: could not reach the hub at $HUB_URL; maintenance proceeds with the fleet unheld"
        fi
        ;;
    task-begin)
        reason="snapraid $what"; [ -n "$high" ] && reason="$reason ($high)"
        if hold "$reason" "$HOLD_TTL_SECONDS"; then
            say "fleet held for $what (ttl ${HOLD_TTL_SECONDS}s)"
        else
            say "WARNING: could not reach the hub at $HUB_URL; $what proceeds with the fleet unheld"
        fi
        ;;
    task-end|task-error)
        if hold "snapraid $what finished; covering spin-down" "$HOLD_TAIL_SECONDS"; then
            say "fleet hold extended ${HOLD_TAIL_SECONDS}s past $what ($event) for spin-down"
        else
            say "WARNING: could not reach the hub at $HUB_URL after $what; the hold lapses on its own"
        fi
        ;;
    *)
        say "unknown event '$event'; ignoring"
        ;;
esac

exit 0

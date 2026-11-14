# busy(): another a94 / a82four / a96 SUMO job on this box -> 0 (busy). Narrow patterns (own pid excluded):
#   [/]root/a94/run_a94\.sh | [a]94queue\.sh             (A94/A95 60 % panel chain and its queue)
#   [/]root/a82four/run_a82\.sh                          (A82 part B four-arm chain)
#   [/]root/a96[a-z0-9_]*/run_a96\.sh                    (any A96 root's stage script, incl. other agents' a96p* roots)
#   [a]96q_[a-z0-9_]*\.sh                                (any other A96 queue script)
#   + /root/a82four/chain_confirm.out without A82F_CONFIRM_DONE or a FAILED line
busy() {
  local P F=/root/a82four/chain_confirm.out B=""
  P=$(ps -eo pid=,args= | awk -v me=$$ '$1 != me' | grep -E '[/]root/a94/run_a94\.sh|[a]94queue\.sh|[/]root/a82four/run_a82\.sh|[/]root/a96[a-z0-9_]*/run_a96\.sh|[a]96q_[a-z0-9_]*\.sh' | cut -c1-160)
  [ -n "$P" ] && B="process: $(echo "$P" | head -3 | tr '\n' ';')"
  if [ -f $F ] && ! grep -qE 'A82F_CONFIRM_DONE|FAILED' $F; then B="$B $F has no finish marker"; fi
  [ -n "$B" ] && { echo "$(date '+%F %T') BUSY $B" >> $L; return 0; }
  return 1
}

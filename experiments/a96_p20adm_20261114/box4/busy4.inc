# busy() (Box 4): another SUMO job on this box -> 0 (busy). Narrow patterns (lines naming this queue script excluded):
#   [g]o_train\.sh | [w]ms\.control\.mappo\.train        (A77 MAPPO20C training chain / trainer, incl. sanity_post)
#   [/]root/a96[a-z0-9_]*/run_a96\.sh | [a]96q_[a-z0-9_]*\.sh   (any A96 stage script / other A96 queue)
#   [/]root/a6[89]/run_a6[89]\.sh                        (A68 / A69 chains present on this box)
busy() {
  local P B=""
  P=$(ps -eo pid=,args= | awk -v me="$(basename "$0")" 'index($0, me) == 0' | grep -E '[g]o_train\.sh|[w]ms\.control\.mappo\.train|[/]root/a96[a-z0-9_]*/run_a96\.sh|[a]96q_[a-z0-9_]*\.sh|[/]root/a6[89]/run_a6[89]\.sh' | cut -c1-160)
  [ -n "$P" ] && B="process: $(echo "$P" | head -3 | tr '\n' ';')"
  [ -n "$B" ] && { echo "$(date '+%F %T') BUSY $B" >> $L; return 0; }
  return 1
}

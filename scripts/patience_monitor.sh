#!/bin/bash
# Manual early-stopping (patience=3) for the two real-scale training arms, then
# wait for the afterany-chained evals + summary and print the final comparison.
cd /home/vkshirsa/srs-styled-map2map-galaxy
source /home/vkshirsa/venv/bin/activate 2>/dev/null
A=8407065; B=8413515
read EA EB SUM < .realchain.txt
PAT=3
LOGA=logs/${A}_patchreal.log; LOGB=logs/${B}_patchreal.log

# latest_epoch best_epoch best_val  (min val over all epochs = best, since best.pt tracks min)
parse(){ grep -E "epoch [0-9]+ " "$1" 2>/dev/null \
  | sed -E 's/.*epoch ([0-9]+).*val_pkRMS=([0-9.]+).*/\1 \2/' \
  | awk 'NF==2{e=$1;v=$2+0; if(b==""||v<b){b=v;be=e}; if(e>le)le=e} END{if(le=="")print "NA"; else print le,be,b}'; }
running(){ squeue -h -j "$1" -o "%T" 2>/dev/null | grep -q RUNNING; }

echo "=== patience monitor start (PAT=$PAT epochs); A=$A B=$B ==="
while running "$A" || running "$B"; do
  for pair in "$A:$LOGA:ArmA-naive" "$B:$LOGB:ArmB-overlap"; do
    job=${pair%%:*}; tmp=${pair#*:}; log=${tmp%%:*}; name=${tmp#*:}
    running "$job" || continue
    r=$(parse "$log"); [ "$r" = "NA" ] && continue
    set -- $r; le=$1; be=$2; bv=$3; gap=$((le-be))
    if [ "$gap" -ge "$PAT" ]; then
      echo "$(date +%H:%M) $name: latest ep$le, best ep$be ($bv) -> flat $gap>=$PAT epochs, scancel $job"
      scancel "$job"
    else
      echo "$(date +%H:%M) $name: latest ep$le, best ep$be ($bv), gap=$gap (<$PAT, keep training)"
    fi
  done
  sleep 1500
done
echo "=== both training jobs terminal; evals (afterany) should now run ==="

# wait for the summary job to finish (it is afterany-chained on both evals)
while true; do
  S=$(sacct -j "$SUM" --format=State -n 2>/dev/null | head -1 | tr -d ' ')
  echo "$(date +%H:%M) summary($SUM)=$S evalA=$(sacct -j $EA --format=State -n 2>/dev/null|head -1|tr -d ' ') evalB=$(sacct -j $EB --format=State -n 2>/dev/null|head -1|tr -d ' ')"
  echo "$S" | grep -qE "COMPLETED|FAILED|TIMEOUT|CANCELLED" && break
  sleep 600
done

echo "=== FINAL: summary log ==="
ls -t logs/*_sumreal.log 2>/dev/null | head -1 | xargs -r cat
echo "=== seam A ==="; cat runs/patch_real/seam_realA_naive.md 2>/dev/null
echo "=== seam B ==="; cat runs/patch_real/seam_realB_overlap.md 2>/dev/null
echo "=== done ==="

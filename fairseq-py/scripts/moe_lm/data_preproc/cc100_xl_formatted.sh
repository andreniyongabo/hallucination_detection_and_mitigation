tgt=/large_experiments/moe/cc100_xl/bin
src=/large_experiments/flores/namangoyal/cc100_combined/bin
mkdir -p $tgt
for lang in en_XX vi_VN ru_RU de_DE fr_XX es_XX bg_BG el_GR ar_AR tr_TR th_TH hi_IN ur_PK sw_KE zh_CN zh_TW ; do
    if [ $lang == "ur_PK" ] || [ $lang == "sw_KE" ] || [ $lang == "zh_TW" ] ; then
        last_shard=7
    else
        last_shard=63
    fi
    echo "$lang $last_shard"
    #du -chs "/large_experiments/flores/namangoyal/cc100_combined/bin/$lang"
    for j in $(seq 0 63); do
        if [ ${last_shard} == 7 ] ; then
            i=$(( j % 8 ))
        else
            i=$j
        fi
        sdir="$tgt/shard$j/$lang"
        mkdir -p $sdir
        ln -s $src/$lang/shard$i/train.* $sdir/
        ln -s $src/$lang/shard$i/dict.txt $sdir/
        ln -s $src/$lang/valid/valid.* $sdir/
        if [ $lang == "en_XX" ] ; then
            ln -s $src/$lang/shard$i/dict.txt $tgt/shard$j/
        fi
    done
done

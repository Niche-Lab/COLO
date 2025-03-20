# for t in {0..3}
# do
#     sh study1.sh $t &
# done

for i in {0..10}
do
    python study1.py -i $i -t $1
done

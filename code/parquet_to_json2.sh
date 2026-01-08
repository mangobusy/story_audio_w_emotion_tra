for id in $(seq $1 $2)
do
    python parquet_to_json2.py $id > "logs/${id}.log" 2>&1 &
done

wait
echo "All processes are complete."


# mkdir logs
# sh parquet_to_json2.sh 0 17
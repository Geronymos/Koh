for i in {0..100}
do
    curl "https://thispersondoesnotexist.com/image" > "dataset/image_$i.png"
    echo "Downloading image number $i"
    sleep 1.2
done
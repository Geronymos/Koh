for file in dataset/kaggle/*
do
    convert "$file" -resize 800x800 -background black -gravity center -extent 800x800 "$file"
done 

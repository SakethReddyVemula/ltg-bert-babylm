mkdir -p ../data/processed_100M

python3 aochildes.py 100M
python3 bnc_spoken.py 100M
python3 cbt.py 100M
python3 children_stories.py 100M
python3 gutenberg.py 100M
python3 open_subtitles.py 100M
python3 qed.py 100M
python3 simple_wikipedia.py 100M
python3 switchboard.py 100M
python3 wikipedia.py 100M

cat ../data/processed_100M/aochildes.txt ../data/processed_100M/bnc_spoken.txt ../data/processed_100M/cbt.txt ../data/processed_100M/children_stories.txt ../data/processed_100M/gutenberg.txt ../data/processed_100M/open_subtitles.txt ../data/processed_100M/qed.txt ../data/processed_100M/simple_wikipedia.txt ../data/processed_100M/switchboard.txt ../data/processed_100M/wikipedia.txt > ../data/processed_100M/all.txt

python3 segment.py 100M

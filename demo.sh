#!/bin/bash


# first _you_ need to install the dependencies for python
# this can be done with `pip install -r requirements.txt`

# set exit on error and print commands
set -xe

PYTHON=`which python2`
WIKIPEDIA_DUMP='enwiki-latest-pages-articles.xml'
WIKIPEDIA_DOWNLOAD='https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2'
WIKIPEDIA_W2V='enwiki-vecs'


mkdir -p other_projects
mkdir -p data

# try running the program first just to make sure that it should be somewhat installed
$PYTHON -c 'import runner' 2>&1 > /dev/null

# first download the needed files
# wikipedia dump etc

if [ ! -f data/$WIKIPEDIA_DUMP ] ; then
	# we used the 20141208 dump so there will be some minor differences
	curl $WIKIPEDIA_DOWNLOAD | bzip2 -d > data/$WIKIPEDIA_DUMP
fi

if [ ! -d ./other_projects/berkeley-entity ] ; then
	SBT='sbt'
	set +e
	which sbt 2>&1 > /dev/null
	if [ $? -ne 0 ] ; then
		if [ ! -f ./other_projects/sbt-launch.jar ] ; then
			curl -o ./other_projects/sbt-launch.jar http://static.matthewfl.com/downloads/sbt-launch.jar
		fi
		SBT="java -Xmx1536M -Xms512M -XX:+CMSClassUnloadingEnabled -jar `pwd`/other_projects/sbt-launch.jar"
	fi
	set -e
	cd other_projects
	git clone https://github.com/matthewfl/berkeley-entity.git
	cd berkeley-entity
	git checkout release
	$SBT assembly
	if [ ! -d ./models ] ; then
		curl http://nlp.cs.berkeley.edu/downloads/berkeley-entity-models.tgz | tar xzf -
	fi
	cd ../..
fi

# run the word2vec steps
if [ ! -f "data/$WIKIPEDIA_W2V.bin" ] ; then
	# process the wikipedia file into files for the word2vec framework
	python wikireader.py data/$WIKIPEDIA_DUMP "data/$WIKIPEDIA_W2V-redir.json" "data/$WIKIPEDIA_W2V-surface.json" "data/$WIKIPEDIA_W2V.txt"
	cd other_projects
	# just a copy of the standard word2vec framework
	git clone https://github.com/matthewfl/word2vec.git word2vec
	cd word2vec
	make
	cd ../..
	./other_projects/word2vec/word2vec -train "data/$WIKIPEDIA_W2V.txt" -output "data/$WIKIPEDIA_W2V.bin" -binary 1 -threads 8 -window 21 -negative 10 -size 300
fi

# download a dataset
if [ ! -d ./data/data/ace05 ] ; then
	cd data
	curl http://static.matthewfl.com/downloads/2015-elink-ace-data.tgz | tar xzf -
	cd ..
fi

BENTITY="java -cp `pwd`/other_projects/berkeley-entity/target/scala-2.11/berkeley-entity-assembly-1.jar -XX:+AggressiveOpts -XX:+UseLargePages -XX:+UseStringDeduplication"

# generate the "wikipedia interface" that is used by berkeley-entity
if [ ! -f ./data/wikipedia-interface.ser.gz ] ; then
	# this isn't needed, but it will save a bunch of time
	if [ ! -f ./data/wikipedia-category-cache.ser.gz ] ; then
		curl -o ./data/wikipedia-category-cache.ser.gz http://static.matthewfl.com/downloads/2015-elink-wikipedia-category-cache.ser.gz
	fi
	DP=`pwd`/data
	cd other_projects/berkeley-entity/
	$BENTITY -Xmx20g edu.berkeley.nlp.entity.wiki.WikipediaInterface -wikipediaDumpPath ../../data/$WIKIPEDIA_DUMP -categoryDBInputPath ../../data/wikipedia-category-cache.ser.gz -outputPath ../../data/wikipedia-interface.ser.gz -mentionType ace -datasetPaths $DP/data/ace05/dev/,$DP/data/ace05/test/,$DP/data/ace05/train/ -wikiStandoff $DP/data/ace05/ace05-all-conll-wiki/
	cd ../..
fi

if [ ! -f ./data/ace-dev-set-queries.json ] ; then
	DP=`pwd`/data
	cd other_projects/berkeley-entity/
	$BENTITY -Xmx20g edu.berkeley.nlp.entity.wiki.JointQueryDenotationChooser -wikiDBPath ../../data/wikipedia-interface.ser.gz -wikiPath $DP/data/ace05/ace05-all-conll-wiki/ -trainDataPath $DP/data/ace05/train/ -testDataPath $DP/data/ace05/dev/ -saveExternalWikiProcess ../../data/ace-dev-set-queries.json
	cd ../..
fi


# finally

python runner.py --queries ./data/ace-dev-set-queries.json --surface_count ./data/enwiki-vecs-surface.json --wordvecs ./data/enwiki-vecs.bin --redirects ./data/enwiki-vecs-redir.json --wiki_dump ./data/enwiki-latest-pages-articles.xml --raw_output ./data/ace-dev-set.h5 --csv_output ./data/ace-dev-set.csv

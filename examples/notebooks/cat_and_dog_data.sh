# download tess data
# check if first argument has been set 
if [ ! -z "$1" ]; then
    DATA_DIR=$1
else
    DATA_DIR=data
fi

echo "Downloading data to $DATA_DIR"

if [ ! -d "$DATA_DIR/CATDOG" ]; then
    kaggle datasets download tongpython/cat-and-dog -p $DATA_DIR/CATDOG --unzip
fi
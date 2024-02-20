# download tess data
# check if first argument has been set 
if [ ! -z "$1" ]; then
    DATA_DIR=$1/data
else
    DATA_DIR=data
fi

echo "Downloading data to $DATA_DIR"

if [ ! -d "$DATA_DIR/TESS" ]; then
    kaggle datasets download ejlok1/toronto-emotional-speech-set-tess -p $DATA_DIR --unzip
    mv "$DATA_DIR/TESS Toronto emotional speech set data" $DATA_DIR/TESS
    rm -r "$DATA_DIR/TESS/TESS Toronto emotional speech set data"
fi
if [ ! -d "$DATA_DIR/RAVDESS_SONG" ]; then
    kaggle datasets download uwrfkaggler/ravdess-emotional-song-audio -p $DATA_DIR/RAVDESS_SONG --unzip
fi
if [ ! -d "$DATA_DIR/RAVDESS_SPEECH" ]; then
    kaggle datasets download uwrfkaggler/ravdess-emotional-speech-audio -p $DATA_DIR/RAVDESS_SPEECH --unzip
fi
if [ ! -d "$DATA_DIR/CREMA-D" ]; then
    kaggle datasets download ejlok1/cremad -p $DATA_DIR --unzip
    mv $DATA_DIR/AudioWAV $DATA_DIR/CREMA-D
fi
if [ ! -d "$DATA_DIR/SAVEE" ]; then
    kaggle datasets download ejlok1/surrey-audiovisual-expressed-emotion-savee -p $DATA_DIR --unzip
    mv $DATA_DIR/ALL $DATA_DIR/SAVEE
fi
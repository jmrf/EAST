

MODEL_PATH="checkpoints/east_icdar2015_resnet_v1_50_rbox"


if [ ! -d ${MODEL_PATH} ] ; then
  echo "Downloading pre-trained model file"
    python ./scripts/download_model.py 11n1Ccs9-XRaijdsgkvVbUXnZnAbUXWd0 "${MODEL_PATH}.zip"

    echo "Unpacking..."
    unzip "${MODEL_PATH}.zip" -d checkpoints

    echo "Removing intermediate files..."
    rm "${MODEL_PATH}.zip"

    echo "Done!"

else
    echo "model checkpoint already present"
fi


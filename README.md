# Generative Machine Learning on the Cloud

This tool uses the [Google Cloud Machine Learning
API](https://cloud.google.com/ml) and [Tensorflow](https://tensorflow.org).

Generative Machine Learning on the Cloud is a cloud based tool to aid in
generative art and synthetic image generation. The end to end system design
allows a user to have a custom dataset of images to train a Variational
Autoencoder Generative Adversarial Network (VAE-GAN) model on Cloud ML. From
here, their model is deployed to the cloud, where they can input an embedding to
have synthetic images generated from their dataset or input an image to get an
embedding vector.

## How To Use the Tool

### Pre-steps:

1.  [Install Tensorflow](https://www.tensorflow.org/install/)
    *   Really recommend doing the virtualenv install
    *   Verify numpy is installed
2.  [Set Up the Cloud
    Environment](https://cloud.google.com/ml-engine/docs/quickstarts/command-line)
    *   Create a Cloud Platform Project
    *   Enable Billing
    *   Enable Cloud ML Engine and Compute Engine APIs
3.  Clone this repo
    *   If using a TensorFlow virtualenv, make sure to clone into a subdirectory
        of the virtualenv directory

### How To: Run a Training Job

A training job will train the VAE-GAN on your own training data!

Warning: You will be using billable components of the Cloud Platform and will
incur charges when running a training job.

1.  cd into the data directory of the source code you just cloned. Make sure to
    activate the tensorflow virtualenv (if that is the method you chose to
    install TensorFlow).

2.  Run the training script \
    \
    Dataset Tips:

    *   Read [how Cloud ML interacts with the
        data](https://cloud.google.com/ml-engine/docs/how-tos/working-with-data).
    *   Accepted image formats: .jpg or .png
    *   The larger your image set, the less chance of overfitting!
    *   One rule of thumb is at least ~1000 images per class.
        -   If you are trying to synthesize faces, try to have at least 1000
            face images.
        -   If you are trying to generate both cat and dog images, try to have
            at least 1000 cats and 1000 dogs.
    *   The model will crop / resize your images to 64x64 squares.
        -   Use the -c flag to specify centered cropping (or else it will random
            crop).
        -   The image is cropped to a bounding box of side lengths of 0.75 *
            minimum(original_height, original_width).
        -   The image is resized to 64x64 (using
            [tf.image.resize_images](https://www.tensorflow.org/api_docs/python/tf/image/resize_images)
            to either downsample or upsample using bilinear interpolation).
    *   This script will turn your image files into [TFRecords file
        format](https://www.tensorflow.org/versions/r1.0/api_guides/python/python_io)
        with Example protos and saves them to your GCS bucket. It partitions
        your data into a training dataset and a validation dataset.
    *   For efficient throughput, image files should not exceed 4 MB. Reducing
        image size can increase throughput.

    Example:

    ```shell
    sh run_training.sh -d $PATH_TO_TRAINING_DIR -c
    ```

    **Flags:** \
    \[-d PATH_TO_TRAINING_DIR\] : required, supplies image directory of .jpg or
    .png images \
    \[-c\] : optional, if present images will be center-cropped, if absent
    images will be randomly cropped. \
    \[-p\] : optional, port on which to start TensorBoard instance.

3.  Monitor your training job using the TensorBoard you started or the Cloud
    dashboard

    *   TensorBoard: Starts at http://0.0.0.0:6006 by default, unless port
        specified.
    *   Job Logs: http://console.cloud.google.com -> Big Data -> ML Engine ->
        Jobs

### How To: Create and Deploy Model

Now that we have a trained model saved on GCS, lets deploy it on Cloud ML!

1.  cd into the data directory of the source code.
2.  Run create model script (if you don't know your job name, use the -l flag) \
    Example:

    ```shell
     sh create_model.sh -j $JOB_NAME
    ```

    **Flags:** \
    \[-j JOB_NAME\] : required unless -l flag present, supplies job name \
    \[-l\]: optional, if present lists 10 most recent jobs created by user

3.  Look at your deployed model on the cloud dashboard under Cloud ML Engine!

    *   Model: http://console.cloud.google.com -> Big Data -> ML Engine ->
        Models

### How To: Run an Inference Job

Now that we have a deployed model trained with your own data, we can use it to
generate new samples.

1.  Generate an Image!

    *   I've provided a script to randomly generate an image from your model and
        display it:

        ```shell
         sh generate_image.sh -m $MODEL_NAME
        ```

        **Flags:** \
        \[-n MODEL_NAME\] : required unless -l flag present, specifies model to
        generate image. \
        \[-l\] : optional, if present lists all models associated with user. \
        \[-d TEMP_DIR\] : optional, directory to which to write json file.

    *   Assumes [PIL is installed](https://pypi.python.org/pypi/Pillow/2.2.1)

        ```shell
        $pip install Pillow
        ```

2.  Embedding to Image generation

    *   Use the command line & a json file!

        *   Example format:

            ```json
            json format:
            {"embeddings": [5,10,-1.6,...,7.8], "key": "0"}
            ```

        *   Embedding array must have dimension of 100 (if using current
            vae-gan) or whatever was specified in the code:

            ```python
            model.py:32 EMBEDDING_SIZE = 100
            ```

        *   Example command:

            ```shell
            gcloud ml-engine predict --model $MODEL_NAME --json-instances $JSON_FILE
            ```

    *   Batch Prediction Job

        *   Example format:

            ```json
            json format example:

            {"embeddings": [0.1,2.3,-4.6,6.5,...,0,4.4,-0.9,-0.9,2.2], "key": "0"}
            {"embeddings": [0.1,2.3,-4.6,6.5,...,1,4.4,-0.9,-0.9,2.2], "key": "1"}
            {"embeddings": [0.1,2.3,-4.6,6.5,...,2,4.4,-0.9,-0.9,2.2], "key": "2"}
            {"embeddings": [0.1,2.3,-4.6,6.5,...,3,4.4,-0.9,-0.9,2.2], "key": "3"}
            {"embeddings": [0.1,2.3,-4.6,6.5,...,4,4.4,-0.9,-0.9,2.2], "key": "4"}
            {"embeddings": [0.1,2.3,-4.6,6.5,...,5,4.4,-0.9,-0.9,2.2], "key": "5"}
            ```

        *   Json file must be on GCS

        *   Example command:

            ```shell
            gcloud ml-engine jobs submit prediction $JOB_NAME --model
            $MODEL_NAME --input-paths "gs://BUCKET/request.json" --output-path
            "gs://BUCKET/output" --region us-east1 --data-format "TEXT"
            ```

    *   Use python API

        *   Documentation
            [here](https://cloud.google.com/ml-engine/docs/tutorials/python-guide)

        *   Setup project and execute request

            ```python
            credentials = GoogleCredentials.get_application_default()
            ml = discovery.build('ml', 'v1', credentials=credentials)
            request_dict = {'instances': [{'embeddings': embeds.tolist(), 'key': '0'}]}
            request = ml.projects().predict(name=model_name, body=request_dict)
            response_image = request.execute()
            ```

3.  Image to Embedding generation

    *   Use the command line & a json file!

        *   Image has to be base64 encoded jpeg
        *   Example format:

            ```json
            json format:
            {"image_bytes": {"b64":"/9j/4AAQSkZJAAQABX...zW0=="}, "key": "0"}
            ```

    *   Batch Prediction

        *   Same as for embedding to image, but with image format json

    *   Python API

        *   Same as for embedding to image, but request_dict:

            ```python
            request_dict = {'instances': [{'image_bytes': {'b64': img}, 'key': '0'}]}
            ```

            Where img is a base64 encoded jpeg

## Acknowledgements

Huge shoutout to this awesome
[DCGAN](https://github.com/carpedm20/DCGAN-tensorflow). After much trial error,
the architecture from this network was the one that produced the greatest
generative results and ended up as the network architecture in the final version
of this tool.

## Disclaimer

This is not an official Google product.

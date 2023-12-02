def predict():
    try:
        # Get the image data from the request
        data_url = request.form['image']
        header, encoded = data_url.split(",", 1)
        image_data = BytesIO(base64.b64decode(encoded))

        # Read the image
        img = image.load_img(image_data, target_size=(28, 28), color_mode="grayscale")

        # Convert the image to a numpy array
        img_array = image.img_to_array(img)
        img_array = np.expand_dims(img_array, axis=0) / 255.0  

        # Make a prediction
        prediction = model.predict(img_array)

        # Get the predicted digit
        predicted_digit = np.argmax(prediction)

        return jsonify({'result': str(predicted_digit)})
    
    except Exception as e:
        print("Error:", str(e))
        return jsonify({'error': str(e)})
def window_realized_handler(app, widget):
    # Hide the image input when the window is realized
    app.input_image_wrapper.hide()
    app.mask_image_wrapper.hide()
    app.image_input.hide()
    app.mask_image.hide()
    app.copy_image_button.hide()

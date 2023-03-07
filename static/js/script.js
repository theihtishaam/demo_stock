// This script is used to add interactivity to the web pages

// When the document is ready, do the following
$(document).ready(function() {
    
    // When the form is submitted, do the following
    $("#predict-form").submit(function(event) {
        // Prevent the default form submission behavior
        event.preventDefault();
        
        // Get the form data
        var form_data = $(this).serialize();
        
        // Send an AJAX request to the server with the form data
        $.ajax({
            url: "predict",
            type: "POST",
            data: form_data,
            success: function(response) {
                // When the server responds with the prediction, update the result element
                $("#result").text("The predicted price is " + response);
            },
            error: function() {
                // If there was an error with the AJAX request, display an error message
                $("#result").text("There was an error with the prediction. Please try again later.");
            }
        });
    });
    
});


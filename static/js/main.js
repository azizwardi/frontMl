// JavaScript for Project Delay Prediction app

document.addEventListener('DOMContentLoaded', function() {
    // Form validation for registration
    const registerForm = document.querySelector('form[action="/register"]');
    if (registerForm) {
        registerForm.addEventListener('submit', function(event) {
            const password = document.getElementById('password').value;
            const confirmPassword = document.getElementById('confirm_password').value;

            if (password !== confirmPassword) {
                event.preventDefault();
                alert('Passwords do not match!');
            }
        });
    }

    // Form validation for prediction
    const predictionForm = document.querySelector('form[action="/prediction"]');
    if (predictionForm) {
        predictionForm.addEventListener('submit', function(event) {
            const hoursSpent = document.getElementById('hours_spent').value;
            const progress = document.getElementById('progress').value;

            if (hoursSpent < 0 || hoursSpent > 100) {
                event.preventDefault();
                alert('Hours spent should be between 0 and 100');
            }

            if (progress < 0 || progress > 1) {
                event.preventDefault();
                alert('Progress should be between 0 and 1');
            }
        });
    }

    // Flash messages will stay on screen (auto-hide disabled)
    const flashMessages = document.querySelectorAll('.alert');
    if (flashMessages.length > 0) {
        // Add close buttons to flash messages so users can manually dismiss them
        flashMessages.forEach(function(message) {
            // Only add close button if it doesn't already have one
            if (!message.querySelector('.btn-close')) {
                const closeButton = document.createElement('button');
                closeButton.className = 'btn-close';
                closeButton.setAttribute('type', 'button');
                closeButton.setAttribute('aria-label', 'Close');
                closeButton.style.float = 'right';

                closeButton.addEventListener('click', function() {
                    message.remove();
                });

                message.prepend(closeButton);
            }
        });
    }
});

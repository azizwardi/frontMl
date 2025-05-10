// Remove all animations from modals and other elements
document.addEventListener('DOMContentLoaded', function() {
    console.log('No animations script loaded');
    
    // Function to remove animations from modals
    function removeModalAnimations() {
        // Remove fade class from all modals
        const modals = document.querySelectorAll('.modal');
        modals.forEach(function(modal) {
            modal.classList.remove('fade');
            
            // Also remove fade from modal dialog
            const dialog = modal.querySelector('.modal-dialog');
            if (dialog) {
                dialog.style.transform = 'none';
                dialog.style.transition = 'none';
            }
        });
        
        // Remove fade class from all backdrops
        const backdrops = document.querySelectorAll('.modal-backdrop');
        backdrops.forEach(function(backdrop) {
            backdrop.classList.remove('fade');
            backdrop.style.opacity = '0.5';
            backdrop.style.transition = 'none';
        });
    }
    
    // Run immediately
    removeModalAnimations();
    
    // Also run when new content might be added to the DOM
    const observer = new MutationObserver(function(mutations) {
        removeModalAnimations();
    });
    
    // Start observing the document with the configured parameters
    observer.observe(document.body, { childList: true, subtree: true });
    
    // Override Bootstrap's Modal show method to prevent animations
    if (typeof bootstrap !== 'undefined' && bootstrap.Modal) {
        const originalShow = bootstrap.Modal.prototype.show;
        bootstrap.Modal.prototype.show = function() {
            // Remove fade class before showing
            if (this._element) {
                this._element.classList.remove('fade');
                
                const dialog = this._element.querySelector('.modal-dialog');
                if (dialog) {
                    dialog.style.transform = 'none';
                    dialog.style.transition = 'none';
                }
            }
            
            // Call the original method
            originalShow.apply(this, arguments);
            
            // Make sure backdrop doesn't animate
            setTimeout(function() {
                const backdrops = document.querySelectorAll('.modal-backdrop');
                backdrops.forEach(function(backdrop) {
                    backdrop.classList.remove('fade');
                    backdrop.style.opacity = '0.5';
                    backdrop.style.transition = 'none';
                });
            }, 0);
        };
    }
    
    // Add a style element to override animations
    const style = document.createElement('style');
    style.textContent = `
        .modal, .modal-backdrop, .modal-dialog {
            transition: none !important;
            animation: none !important;
        }
        
        .modal.fade .modal-dialog {
            transform: none !important;
        }
        
        .modal-backdrop.fade {
            opacity: 0.5 !important;
        }
        
        .card:hover, .btn:hover, .dashboard-card:hover {
            transform: none !important;
            box-shadow: none !important;
        }
    `;
    document.head.appendChild(style);
});

function validateForm() {
    // Get form elements
    const name = document.getElementById('name').value;
    const email = document.getElementById('email').value;
    const password = document.getElementById('password').value;
    const confirmPassword = document.getElementById('confirmPassword').value;
  
    // Regular expressions for validation
    const nameRegex = /^[a-zA-Z ]+$/;
    const emailRegex = /^[^\s@]+@[^\s@]+\.[^\s@]+$/;
    const passwordRegex = /^(?=.[A-Za-z])(?=.\d)[A-Za-z\d]{8,}$/;
  
    // Validation logic
    if (name === '') {
      alert('Please enter your name.');
      return false;
    } else if (!nameRegex.test(name)) {
      alert('Please enter a valid name.');
      return false;
    }
  
    if (email === '') {
      alert('Please enter your email address.');
      return false;
    } else if (!emailRegex.test(email)) {
      alert('Please enter a valid email address.');
      return false;
    }
  
    if (password === '') {
      alert('Please enter your password.');
      return false;
    } else if (!passwordRegex.test(password)) {
      alert('Password must be at least 8 characters long and contain at least one letter and one number.');
      return false;
    }
  
    if (confirmPassword === '') {
      alert('Please confirm your password.');
      return false;
    } else if (confirmPassword !== password) {
      alert('Passwords do not match.');
      return false;
    }
  
    // If all validations pass, submit the form
    return true;
  }


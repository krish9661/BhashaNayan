function validateLogin() {
    const username = document.getElementById('username').value;
    const password = document.getElementById('password').value;
  
    // input validation
    if (username.trim() === '') {
      alert('Please enter a username.');
      return false;
    }
  
    if (password.trim() === '') {
      alert('Please enter a password.');
      return false;
    }
}
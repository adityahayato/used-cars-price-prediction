const themeToggle = document.querySelector('.theme-toggle');
const savedTheme = localStorage.getItem('carrcheck-theme');

function setTheme(isDark) {
  document.body.classList.toggle('dark-theme', isDark);
  if (themeToggle) {
    themeToggle.textContent = isDark ? '☀ Light' : '☾ Dark';
    themeToggle.setAttribute('aria-pressed', String(isDark));
  }
}

setTheme(savedTheme === 'dark');

themeToggle?.addEventListener('click', () => {
  const isDark = !document.body.classList.contains('dark-theme');
  setTheme(isDark);
  localStorage.setItem('carrcheck-theme', isDark ? 'dark' : 'light');
});

document.querySelectorAll('form').forEach(form => form.addEventListener('submit', () => {
  const button = form.querySelector('button[type="submit"]');
  if (button) {
    button.disabled = true;
    button.innerHTML = 'Calculating…';
  }
}));

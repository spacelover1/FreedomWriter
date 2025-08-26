// jalali-calendar.js
// تبدیل تاریخ ISO پست به شمسی و نمایش در span#jalali-date

// تابع کمکی برای تبدیل میلادی به شمسی با Intl
function toPersianDate(gregorian) {
  const date = new Date(gregorian);
  return new Intl.DateTimeFormat('fa-IR', {
    weekday: 'long',
    year: 'numeric',
    month: 'long',
    day: 'numeric'
  }).format(date);
}

document.addEventListener("DOMContentLoaded", function() {
  const el = document.getElementById("jalali-date");
  if(el) {
    const gDate = el.getAttribute("data-gregorian");
    el.innerText = toPersianDate(gDate);
  }
});

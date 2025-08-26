// jalali-calendar.js
// یک تاریخ شمسی ساده برای نمایش
function getPersianDate() {
  var date = new Date();
  var pDate = new Intl.DateTimeFormat('fa-IR', { 
    weekday: 'long', year: 'numeric', month: 'long', day: 'numeric' 
  }).format(date);
  return pDate;
}

document.addEventListener("DOMContentLoaded", function() {
  var el = document.getElementById("jalali-date");
  if(el) {
    el.innerText = getPersianDate();
  }
});

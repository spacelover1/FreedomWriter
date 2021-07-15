---
title: مبانی پردازش زبان طبیعی(NLP)- سه
category: general
tags:    nlp اموزش  
---


در این قسمت در مورد برداری کردن دیتا صحبت می کنیم. <br/>
تا اینجا دیتا رو خوندیم و تا حدودی نرمالیزه کردیم. الان پایتون دیتا رو فقط یک سری رشته کاراکتر می بینه. حالا برای اینکه مدل ماشین لرنینگ و پایتون این دیتا رو درک کنه باید دیتا برداری بشه. برداری کردن یعنی چی؟ یعنی متن به عددصحیح تبدیل شه و یک بردار ویژگی ساخته شه.<br/>
حالا بردار ویژگی در اینجا یعنی متن هر پیام رو بگیریم و به یک بردارعددی تبدیل کنیم که نمایش دهنده متن اون پیام باشه. <br/>
چطوری این کار رو انجام می دیم؟ در ادامه درباره این مورد صحبت می کنیم.<br/>
چندین روش برای برداری کردن ویژگی ها وجود داره که در ادامه سه روش رایج رو بررسی می کنیم.

## روش اول: بردار تعداد (Count Vectorization)

در این روش هر پیام گرفته می شه و هر کلمه به عنوان یک ویژگی در نظر گرفته می شه و بعد تعداد تکرار هر کلمه در اون پیام ثبت می شه. در نهایت یک ماتریسی داریم که هر سطر مربوط به یک پیام و هر ستون نمایش دهنده یک کلمه است. و در نهایت پایتون با بررسی این ماتریس یک ارتباطی بین کلمات موجود در پیام و لیبل اون پیام پیدا می کنه تا در آینده که بهش پیام های بدون لیبل بدیم بتونه به درسی برچسب گذاری کنه.

برای درک بهتر این فرایند به عکس زیر دقت کنید: 

![vectorization_example](https://raw.githubusercontent.com/spacelover1/NLP-with-Python/main/3-VectorizingRawData/vectorization_example.PNG)

دراین تصویر فقط دو رشته offer و lol از لیست کلمات پیام ها انتخاب شده و تعداد تکرارشون محاسبه شده. همونطور که در جدول سمت چپ و راست می بینید پیام هایی که برچسب غیر اسپم دارند در آن ها رشته lol وجود داشته و تکرار شده ولی شامل رشته offer نیستند و برعکس پیام های اسپم اکثرا رشته offer رو شامل می شن. <br/>
این یک مثال بسیار ساده برای درک فرایند و مفهوم بردار تعداد است.

حالا در عمل این روش رو پیاده می کنیم:

    

    import pandas as pd
    import re
    import string
    import nltk

    pd.set_option('display.max_colwidth', 100)
    dataset = pd.read_csv('SMSSpamCollection.tsv', sep='\t')
    dataset.columns = ['label', 'body']

    nltk.download('stopwords')
    stopwords = nltk.corpus.stopwords.words('english')
    ps = nltk.PorterStemmer()
    

    def clean_text(text):
      text = "".join([word.lower() for word in text if word not in string.punctuation])
      tokens = re.split('\W+', text)
      text = [ps.stem(word) for word in tokens if word not in stopwords]
      return text


بعد از خوندن و پاکسازی دیتا، سراغ برداری کردن می ریم.


    from sklearn.feature_extraction import CountVectorizer

    count_vect = CountVectorizer(analyzer=func_name)
    X_counts = count_vect.fit_transform(dataset['body'])

حالا می تونیم با استفاده از `X_counts.shape` تعداد پیام ها و تعداد رشته های منحصر بفرد در این پیام ها رو ببینیم. در این دیتاست 5567 پیام و 8104 رشته منحصر بفرد داریم که همون ویژگی های ما هستند. این اعداد تعداد سطرو و ستون های ماتریس رو نمایش می ده.<br/>
و `count_vect.get_feature_names()` رشته های منحصربفرد رو نمایش می ده.

تابع هایپرپارامترهای دیگه ای هم داره که [اینجا](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.CountVectorizer.html) می تونید دربارشون بخونید.

حالا در اینجا برای یادگیری 20 پیام اول رو  برداری می کنیم:

    sample = dataset[0:20]
    count_vect_sample = CountVectorizer(analyzer=clean_text)
    X_counts_sample = count_vect_sample.fit_transform(dataset['body'])
    
و الان وقتی سایز دیتای نمونه رو ببینیم 192 رشته منحصربفرد داریم. خروجی و کد کامل این بخش رو [اینجا](https://github.com/spacelover1/NLP-with-Python/blob/main/3-VectorizingRawData/CountVectorization.ipynb) می تونید ببینید.






















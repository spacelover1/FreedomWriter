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

## **روش اول: بردار تعداد (Count Vectorization)**

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



## **روش دوم: بردار N-Grams (N-gram vectorizing)**

این روش هم تا حدود زیادی مشابه روش قبلیه و ساختار کدش مشابه اونه. در اینجا هم هر سطر پیام ها هستند ولی هر ستون به جای نمایش یک رشته، ترکیب nتایی از رشته هاست. برای درک بهتر تصویر زیر  رو ببینید: 

[ngrams](https://raw.githubusercontent.com/spacelover1/NLP-with-Python/main/3-VectorizingRawData/ngrams.png?token=AEGZAVTZYIIT2UNASADUQN3A6KJCE)


مثل مرحله قبل دیتا رو می خونیم و بعد باید تابعی بنویسیم که مراحل پاکسازی رو انجام بده. بخش قبل یک لیستی از توکن رو می دادیم به vectorizer اما الان چون می خواد ترکیبی از کلمه ها رو بسازه باید ورودی بهش یک رشته بدیم. پس در آخر باید توکن ها رو مثل یک جمله کنار هم دیگه قرار بدیم و این کار رو با تابع `join()` انجام می دیم:

    def clean_text(text):
      text = "".join([word.lower() for word in text if word not in string.punctuation])
      tokens = re.split('\W+', text)
      text = " ".join([ps.stem(word) for word in tokens if word not in stopwords])
      return text

اگر یادتون باشه خط اول کاراکترها رو یکی یکی بررسی می کرد برای پیدا کردن علائم نگارشی و در اخر با `join()` این کاراکترها رو بهم متصل کردیم. در `join()` دومی قرار کلمه ها کنار هم بیان تا جمله بسازن پس باید یک فاصله بین هر کلمه باشه.

در اینجا هم از CountVectorizer استفاده می کنیم: 

    ngram_vect = CountVectorizer(ngram_range=(2,2))
    X_counts = ngram_vect.fit_transform(dataset['cleaned_text'])

در `ngram_range` مشخص می کنیم که ترکیب چندتایی از کلمه ها بسازه. مثلا (1,3) یعنی همه ترکیبهای یکی، دوتایی و سه تایی. <br/>
همونطور که مشاهده می کنید تعداد فیچرها اینجا خیلی زیاد می شه. نکته ای که باید توجه کنیم اینه که چه زمانی از هر کدوم از این روش ها استفاده کنیم. با توجه به مسئله ممکنه یکی از این روش ها نتیجه بهتری بده. لزوما نمی شه گفت یکی از این روش ها بهتر از دیگری است.

کدهای این بخش در [اینجا](https://github.com/spacelover1/NLP-with-Python/blob/main/3-VectorizingRawData/NGrams.ipynb) قابل مشاهده است.



























---
title: مبانی پردازش زبان طبیعی(NLP)- دو
category: general
tags:    nlp اموزش  
---

در قسمت قبل فایل متنی رو به دو روش آسان و دشوار خوندیم. و پاکسازی دیتا رو توسط سه متد پیش پردازش یاد گرفتیم: حذف علائم نگارشی، توکنایز کردن (جداسازی کلمات)، حذف کلمات بدون معنی. و گفتیم که یک مرحله چهارمی هم برای پاکسازی یا نرمالسازی داده متنی می شه استفاده کرد که شاید همیشه به اندازه مراحل قبل اهمیت نداشته باشه. در این قسمت درباره این مرحله چهارم صحبت می کنیم.


## بخش پنجم: stemming

کاری که stemming انجام می ده اینه که میاد پسوند و پیشوند رو از کلمه حذف می کنه و هر چی باقی موند رو به عنوان خروجی میده، پس ممکنه خروجی حتی کلمه نباشه. خب پس چرا ازش استفاده می کنیم؟ بخاطر سادگی و سرعتی که داره.

اول از همه روی یک سری کلمات این روش رو اجرا می کنیم تا بیشتر باهاش آشنا شیم و بعد روی دیتاستی که در قسمت قبل بررسی کردیم. <br/>
باید اول کتابخونه `nltk` رو ایمپورت کنیم. سپس از  استمر `PorterStemmer()`  استفاده می کنیم. این استمرها برای هر زبانی متفاوته، برای زبان انگلیسی دو استمر `PorterStammer` و `LancasterStammer` وجود داره. اولی رایجتره و سریعتر.


    import nltk
    ps = nltk.PorterStemmer()

با استفاده از `dir(ps)` توابعی که این استمر داره رو می تونیم مشاهده کنیم. تابع `stem` بیشتر استفاده می شه. 

    print(ps.stem('grow'))
    print(ps.stem('growing'))
    print(ps.stem('grows'))

همه این کلمات رو خلاصه می کنه به grow. در مثال بعدی فرق فعل و فاعل رو می دونه.

    print(ps.stem('runs'))  
    print(ps.stem('running'))
    print(ps.stem('runner'))

حالا بریم سراغ دیتاست پیام های اسپم و غیر اسپم.<br/>
ابتدا دیتا رو می خونیم:
    import pandas as pd
    import re
    import string

    nltk.download('stopwords')
    stopword = nltk.corpus.stopwords.words('english')
    pd.set_option('display.max_colwidth', 100)

    dataset = pd.read_csv('SMSSpamCollection.tsv', sep='\t', header=None)
    dataset.columns = ('label', 'body')
    dataset.head()

کتابخونه های مورد نیاز رو هم من همین ابتدا ایمپورت کردم. سپس سه مرحله پاکسازی دیتا رو انجام  می دیم:

    def clean_data(text):
        text = [word for word in text if word not in string.punctuation]
        tokens = re.split('\W+', text)
        text = [word for word in tokens if word not in stopword]
        return text

    dataset['tokenized_text'] = dataset['body'].apply(lambda x: clean_data(x))



    def stemming(tokenized_text):
        text = [ps.stem(word) for word in tokenized_text]
        return text

    dataset['stemmed_text'] = dataset['tokenized_text'].apply(lambda x: stemming(x))










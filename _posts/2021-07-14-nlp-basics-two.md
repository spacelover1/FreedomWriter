---
title: مبانی پردازش زبان طبیعی(NLP)- دو
category: general
tags:    nlp اموزش  
---

در قسمت قبل فایل متنی رو به دو روش آسان و دشوار خوندیم. و پاکسازی دیتا رو توسط سه متد پیش پردازش یاد گرفتیم: حذف علائم نگارشی، توکنایز کردن (جداسازی کلمات)، حذف کلمات بدون معنی. و گفتیم که یک مرحله چهارمی هم برای پاکسازی یا نرمالسازی داده متنی می شه استفاده کرد که شاید همیشه به اندازه مراحل قبل اهمیت نداشته باشه. در این قسمت درباره این مرحله چهارم صحبت می کنیم.


## **بخش پنجم: Stemming**

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


## **بخش ششم: Lemmatization**


همونطور که دیدیم خروجی `stemming` لزوما کلمه نیست و ممکنه یک چیز بی معنی باشه یا حتی اشتباه. یک متد دیگه `Lemmatization` نام داره که خروجی این روش حتما کلمه ای در دیکشنریه. یعنی معمولا کلمات رو می بره به ریشه شون.

به مثال های زیر توجه کنید. 


    print(ps.stem('meaning')) ==> mean
    print(ps.stem('meanness')) ==> mean

    print(wn.lemmatize('meaning')) ==> meaning
    print(wn.lemmatize('meanness')) ==> meanness


    print(ps.stem('goose')) ==> goos
    print(ps.stem('geese')) ==> gees

    print(wn.lemmatize('goose')) ==> goose
    print(wn.lemmatize('geese')) ==> goose


در واقع stemming رویکرد الگوریتمی داره و فقط با رشته ای که بهش می دیم کار می کنه و فقط پسوند رو حذف می کنه.<br/>
اما lemmatization پیچیده تره و کلمه ای که بهش داده می شه رو در لیست لغات بررسی می کنه و پرداش می کنه و بعد ریشه کلمه رو بر می گردونه مشکلش اینه که اگر کلمه ای که بهش داده شده در لیست لغات نباشه همونو برمی گردونه.

همین اتفاقی که در این مثال ها افتاده، که همونطور که می بینیم خلاصه نکردن بهتر از رشته کلمه اشتباه برگردوندنه.

حالا می خوایم تکنیک lemmatization رو روی دیتاست پیام ها پیاده کنیم. مثل قبل، ابتدا دیتا رو می خونیم و پاکسازی های اولیه رو انجام می دیم و بعد از lemmatizer استفاده می کنیم. <br/>
در اینجا من فقط تابع lemmatizer رو نوشتم. کد کامل رو [اینجا](https://github.com/spacelover1/NLP-with-Python/blob/main/2-SupplementalDataCleaning/UsingaLemmatizer.ipynb) می تونید مشاهده کنید.

برای درک بهتر این دو روش می تونید [این توضیح انگلیسی](https://www.datacamp.com/community/tutorials/stemming-lemmatization-python) رو مطالعه کنید.





































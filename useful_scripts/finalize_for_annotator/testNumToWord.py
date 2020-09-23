from fa import convert
import re

text = re.sub(r"(\d+)", lambda x : convert(int(x.group(0))), "من سال ۱۴۲۰ تعداد جمعیت با کاهش مواجه خواهد شد  به ۴۷")


print(text)
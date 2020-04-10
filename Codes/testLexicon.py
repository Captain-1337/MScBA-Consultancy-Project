import pandas as pd
from corpora_utils import CorporaHelper,CorporaDomains
from senticnet_utils import SenticNetHelper

"""
corpora_helper = CorporaHelper("multigenre.csv")

blog_corpora = corpora_helper.get_domain_data(CorporaDomains.MOVIEREVIEW.value)
print(CorporaDomains.BLOG.value)
print(blog_corpora)

"""

snh = SenticNetHelper()
emotion = snh.get_emotion('roger')

print(emotion)
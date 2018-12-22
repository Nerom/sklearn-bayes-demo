from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import CountVectorizer

a = [
    {"a": 2, "b": 3},
    {"a": 1, "c": 1, "d": 3}
]
b = [
    {"a": 1, "e": 1}
]
dv = DictVectorizer()
ax = dv.fit_transform(a).toarray()
bx = dv.transform(b).toarray()

print("DictVectorizer:")
print(dv.get_feature_names())
print(ax)
print(bx)
print("\n\n")

c = [
    "a,a,b,b,c,d",
    "a,b,a,c,c"
]
d = [
    "a,e"
]
cv = CountVectorizer(analyzer=lambda s: s.split(","))
cx = cv.fit_transform(c).toarray()
dx = cv.transform(d).toarray()

print("CountVectorizer:")
print(cv.get_feature_names())
print(cx)
print(dx)
print("\n\n")

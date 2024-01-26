from qdrant_client import models, QdrantClient
from sentence_transformers import SentenceTransformer

encoder = SentenceTransformer("all-MiniLM-L6-v2")

documents = [
{
"name": "آلة الزمن",
"description": "رجل يسافر عبر الزمن ويشهد تطور الإنسانية.",
"author": "إتش. جي. ويلز",
"year": 1895,
},
{
"name": "لعبة إندر",
"description": "صبي صغير يتدرب ليصبح قائدًا عسكريًا في حرب ضد جنس فضائي.",
"author": "أورسون سكوت كارد",
"year": 1985,
},
{
"name": "عالم جديد شجاع",
"description": "مجتمع ديستوبي حيث يتم هندسة الناس جينيًا ويتم تحديد سلوكهم للامتثال لتسلسل اجتماعي صارم.",
"author": "ألدوس هكسلي",
"year": 1932,
},
{
"name": "دليل المسافر في مجرة الرمز",
"description": "سلسلة خيال علمي فكاهية تتابع محنة إنسان غير مدرك وصديقه الفضائي.",
"author": "دوغلاس آدمز",
"year": 1979,
},
{
"name": "الكتلة الرملية",
"description": "كوكب صحراوي هو موقع للمكائد السياسية والصراعات السلطوية.",
"author": "فرانك هيربرت",
"year": 1965,
},
{
"name": "المؤسسة",
"description": "عالم الرياضيات يطور علمًا لتنبؤ مستقبل الإنسانية ويعمل على إنقاذ الحضارة من الانهيار.",
"author": "إسحاق أسيموف",
"year": 1951,
},
{
"name": "انهيار الثلج",
"description": "عالم مستقبلي حيث تطور الإنترنت إلى واقع افتراضي متعدد الأبعاد.",
"author": "نيل ستيفنسون",
"year": 1992,
},
{
"name": "نيورومانسر",
"description": "يتم تعيين هاكر لتنفيذ قرصنة تقترب من المستحيل ويتورط في شبكة من التآمر.",
"author": "ويليام جيبسون",
"year": 1984,
},
{
"name": "حرب العوالم",
"description": "غزو مريخي للأرض يلقي البشرية في الفوضى.",
"author": "إتش. جي. ويلز",
"year": 1898,
},
{
"name": "ألعاب الجوع",
"description": "مجتمع ديستوبي حيث يتم إجبار المراهقين على القتال حتى الموت في عرض تلفزيوني مثير.",
"author": "سوزان كولينز",
"year": 2008,
},
{
"name": "فيروس أندروميدا",
"description": "فيروس قاتل من الفضاء الخارجي يهدد بمحو البشرية.",
"author": "مايكل كريتون",
"year": 1969,
},
{
"name": "يد اليسرى للظلام",
"description": "يتم إرسال سفير إنسان إلى كوكب حيث سكانه لا جنس لهم ويمكنهم تغيير جنسهم كما يشاؤون.",
"author": "أورسولا ك. لو غوين",
"year": 1969,
},
{
"name": "مشكلة الجسم الثلاثي",
"description": "البشر يواجهون حضارة فضائية تعيش في نظام متواجد في حالة تدهور.",
"author": "ليو سيشين",
"year": 2008,
},
]


qdrant = QdrantClient(host="localhost",port=6333)
qdrant.recreate_collection(
    collection_name="my_books",
    vectors_config=models.VectorParams(
        size=encoder.get_sentence_embedding_dimension(),  # Vector size is defined by used model
        distance=models.Distance.DOT,
    ),
)

qdrant.upload_records(
    collection_name="my_books",
    records=[
        models.Record(
            id=idx, vector=encoder.encode(doc["description"]).tolist(), payload=doc
        )
        for idx, doc in enumerate(documents)
    ],
)

hits = qdrant.search(
    collection_name="my_books",
    query_vector=encoder.encode("مشكلة").tolist(),
    limit=3,
)
for hit in hits:
    print(hit.payload, "score:", hit.score)
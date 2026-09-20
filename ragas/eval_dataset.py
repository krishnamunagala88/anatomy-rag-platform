"""
Eval question set for the anatomy RAG pipeline.

15 questions spanning different chapters, phrasing styles, and difficulty --
including a few DELIBERATELY cross-referential ones (per our earlier
multi-hop discussion) to see if single-hop retrieval actually struggles with
them or not.

`reference` is a short, independently-written reference answer (not copied
from the book) used by RAGAS's context_recall and answer_correctness-style
metrics to judge whether retrieval/generation captured the right information.
Keep these reference answers accurate but written in your own words.
"""

EVAL_QUESTIONS = [
    {
        "question": "What is circumduction?",
        "reference": "Circumduction is a circular movement of a limb or body "
                      "part, combining flexion, extension, abduction, and "
                      "adduction in sequence, tracing a cone-shaped path."
    },
    {
        "question": "What is the difference between flexion and extension at the neck?",
        "reference": "Flexion at the neck is bending the head forward so the "
                      "face moves closer to the chest. Extension is the "
                      "opposite motion, moving the face away from the chest."
    },
    {
        "question": "What does supination mean?",
        "reference": "Supination is a rotational movement of the forearm that "
                      "turns the palm to face upward or forward."
    },
    {
        "question": "What is the anatomical position?",
        "reference": "The anatomical position is a standard reference posture: "
                      "standing upright, facing forward, arms at the sides "
                      "with palms facing forward, and feet together."
    },
    {
        "question": "Define abduction and adduction of a limb.",
        "reference": "Abduction is movement of a limb away from the midline of "
                      "the body. Adduction is movement of a limb toward the "
                      "midline."
    },
    {
        "question": "What is a synovial joint?",
        "reference": "A synovial joint is a freely movable joint characterized "
                      "by a joint cavity filled with synovial fluid, articular "
                      "cartilage covering the bone ends, and a surrounding "
                      "joint capsule."
    },
    {
        "question": "What are the main types of muscle based on shape?",
        "reference": "Muscles are classified by shape into types such as "
                      "fusiform, triangular, unipennate, bipennate, and "
                      "multipennate, among others."
    },
    {
        "question": "What is the function of cartilage compared to bone?",
        "reference": "Cartilage is a firm but flexible connective tissue that "
                      "cushions and allows some flexibility at joints, whereas "
                      "bone is hard and rigid, providing structural support."
    },
    {
        "question": "What is meant by the term 'proximal' in anatomy?",
        "reference": "Proximal describes a position closer to the trunk or "
                      "point of attachment/origin of a structure, as opposed "
                      "to distal, which means farther away."
    },
    {
        "question": "What is haemorrhage?",
        "reference": "Haemorrhage is bleeding, which can be either external "
                      "(visible, outside the body) or internal (inside body "
                      "cavities or tissues)."
    },
    {
        "question": "What movements are possible in the trunk?",
        "reference": "The trunk can perform flexion (bending forward), "
                      "extension (bending backward), lateral flexion (bending "
                      "sideways), and rotation."
    },
    {
        "question": "What does the term 'ulcer' mean in clinical anatomy?",
        "reference": "An ulcer is a localized break or loss of continuity in "
                      "the surface of skin or mucous membrane."
    },
    {
        # Deliberately cross-referential -- tests whether single-hop
        # retrieval alone surfaces both movement terminology AND joint
        # structure, or whether it needs multi-hop (see earlier chat).
        "question": "How does the structure of a synovial joint make movements "
                    "like flexion and circumduction possible?",
        "reference": "A synovial joint's joint cavity, synovial fluid, and "
                      "articular cartilage allow smooth, low-friction motion "
                      "between bones, which is what permits movements such as "
                      "flexion and circumduction to occur at that joint."
    },
    {
        "question": "What is the difference between pronation and supination?",
        "reference": "Pronation rotates the forearm so the palm faces "
                      "downward or backward; supination rotates it so the "
                      "palm faces upward or forward -- they are opposite "
                      "rotational movements."
    },
    {
        "question": "What are the subdivisions of anatomy as a subject?",
        "reference": "Anatomy is commonly subdivided into areas such as gross "
                      "(macroscopic) anatomy, microscopic anatomy (histology), "
                      "developmental anatomy (embryology), and applied/clinical "
                      "anatomy."
    },
]
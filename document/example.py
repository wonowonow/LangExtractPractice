import os
import textwrap
import langextract as lx
from dotenv import load_dotenv

load_dotenv()

input = """
철수: "영희야, 오늘 날씨가 정말 좋다! 같이 공원에 가서 산책하지 않을래?"
영희: "좋은 생각이야! 하지만 나는 조금 걱정돼. 시험이 다음 주인데..."
철수: "걱정하지 마. 잠깐 휴식을 취하는 것도 공부에 도움이 될 거야."
영희: "고마워, 철수야. 너는 정말 좋은 친구야."
"""

# 1. 프롬프트 및 추출 규칙 정의
prompt = textwrap.dedent("""\
    등장순서대로 인물, 감정, 관계를 추출하세요.
    추출할 때는 원문의 정확한 텍스트를 사용하고, 의역하거나 개체를 중복시키지 마세요.
    각 개체에 대해 맥락을 추가하는 의미 있는 속성을 제공하세요.""")

# 2. 모델을 가이드할 고품질 예제 제공
examples = [
    lx.data.ExampleData(
        text="민수: '안녕 지영아, 오늘 기분이 어때?' 지영: '좋아! 너를 만나서 정말 기뻐.'",
        extractions=[
            lx.data.Extraction(
                extraction_class="인물",
                extraction_text="민수",
                attributes={"역할": "화자", "감정상태": "관심"}
            ),
            lx.data.Extraction(
                extraction_class="인물",
                extraction_text="지영",
                attributes={"역할": "응답자", "감정상태": "기쁨"}
            ),
            lx.data.Extraction(
                extraction_class="감정",
                extraction_text="정말 기뻐",
                attributes={"강도": "높음", "유형": "긍정적"}
            ),
            lx.data.Extraction(
                extraction_class="관계",
                extraction_text="너를 만나서",
                attributes={"유형": "친밀감", "방향": "상호"}
            ),
        ]
    )
]

result = lx.extract(
    text_or_documents=input,
    prompt_description=prompt,
    examples=examples,
    model_id="gpt-4o",
    api_key=os.environ.get('OPENAI_API_KEY'),
    fence_output=True,
    use_schema_constraints=False
)

if __name__ == "__main__": 
    # Save the results to a JSONL file
    lx.io.save_annotated_documents([result], output_name="extraction_results.jsonl", output_dir=".")
    
    # Generate the visualization from the file
    html_content = lx.visualize("extraction_results.jsonl")
    with open("visualization.html", "w") as f:
        if hasattr(html_content, 'data'):
            f.write(html_content.data)  # For Jupyter/Colab
        else:
            f.write(html_content)
    
    print("\nVisualization saved to visualization.html")
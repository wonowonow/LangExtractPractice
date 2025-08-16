import os
import textwrap
import langextract as lx
from dotenv import load_dotenv
import PyPDF2
import fitz  # PyMuPDF

load_dotenv()

def extract_text_from_pdf(pdf_path):
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text()
        doc.close()
    except Exception as e:
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text()
        except Exception as e2:
            print(f"PDF 텍스트 추출 실패: {e2}")
            return None
    
    return text

def extract_tech_stack_from_text(text):
    """텍스트에서 개발자 기술스택을 추출합니다."""
    
    # 기술스택 추출을 위한 프롬프트 정의
    prompt = textwrap.dedent("""\
        개발자 이력서나 문서에서 기술스택을 추출하세요.
        프로그래밍 언어, 프레임워크, 데이터베이스, 도구, 클라우드 서비스 등을 포함합니다.
        정확한 기술명을 사용하고 중복을 피하세요.
        각 기술에 대해 숙련도나 경험 수준을 추정해서 속성으로 제공하세요.
        숙련도를 확인 할 수 없는 경우 "알 수 없음"으로, 확인할 수 있는 경우 고급 중급 초급으로 표시하고, 숙련도 판단 근거를 적으세요.
        """)
    
    # 기술스택 추출을 위한 예제 데이터
    examples = [
        lx.data.ExampleData(
            text="Java와 Spring Boot를 사용한 백엔드 개발 경험 5년. MySQL 데이터베이스 설계 및 최적화. AWS EC2, S3 서비스 활용. React.js로 프론트엔드 개발.",
            extractions=[
                lx.data.Extraction(
                    extraction_class="프로그래밍 언어",
                    extraction_text="Java",
                    attributes={"경험년수": "5년", "숙련도": "고급", "카테고리": "백엔드", "숙련도 판단 근거": "쓰레드 관리, JVM 이해 등 고급 기능 사용 경험 있음"}
                ),
                lx.data.Extraction(
                    extraction_class="프레임워크",
                    extraction_text="Spring Boot",
                    attributes={"분야": "백엔드", "숙련도": "고급", "카테고리": "웹 프레임워크", "숙련도 판단 근거": "REST API 설계 및 마이크로서비스 아키텍처 경험 있음"}
                ),
                lx.data.Extraction(
                    extraction_class="데이터베이스",
                    extraction_text="MySQL",
                    attributes={"유형": "관계형", "숙련도": "중급", "활용영역": "설계 및 최적화", "숙련도 판단 근거": "인덱스 최적화 및 쿼리 튜닝 경험 있음"}
                ),
                lx.data.Extraction(
                    extraction_class="클라우드 서비스",
                    extraction_text="AWS EC2",
                    attributes={"제공업체": "AWS", "서비스유형": "컴퓨팅", "숙련도": "중급", "숙련도 판단 근거": "서버 배포 및 관리 경험 있음"}
                ),
                lx.data.Extraction(
                    extraction_class="클라우드 서비스",
                    extraction_text="AWS S3",
                    attributes={"제공업체": "AWS", "서비스유형": "스토리지", "숙련도": "중급", "숙련도 판단 근거": "파일 저장 및 CDN 활용 경험 있음"}
                ),
                lx.data.Extraction(
                    extraction_class="프레임워크",
                    extraction_text="React.js",
                    attributes={"분야": "프론트엔드", "숙련도": "중급", "카테고리": "UI 라이브러리", "숙련도 판단 근거": "컴포넌트 기반 개발 및 상태 관리 경험 있음"}
                ),
            ]
        )
    ]
    
    # langextract를 사용한 기술스택 추출
    result = lx.extract(
        text_or_documents=text,
        prompt_description=prompt,
        examples=examples,
        model_id="gpt-4o",
        api_key=os.environ.get('OPENAI_API_KEY'),
        fence_output=True,
        use_schema_constraints=False
    )
    
    return result

def main(pdf_path):
    
    # PDF에서 텍스트 추출
    print("PDF에서 텍스트를 추출하는 중...")
    text = extract_text_from_pdf(pdf_path)
    
    if not text:
        print("PDF에서 텍스트를 추출할 수 없습니다.")
        return None
    
    print(f"추출된 텍스트 길이: {len(text)} 문자")
    
    # 기술스택 추출
    print("기술스택을 추출하는 중...")
    result = extract_tech_stack_from_text(text)
    
    return result

if __name__ == "__main__":
    pdf_file_path = "resume.pdf"
    
    if os.path.exists(pdf_file_path):
        result = main(pdf_file_path)
        
        if result:
            lx.io.save_annotated_documents([result], output_name="tech_stack_extraction.jsonl", output_dir=".")
            
            # 시각화 생성
            html_content = lx.visualize("tech_stack_extraction.jsonl")
            with open("tech_stack_visualization.html", "w", encoding='utf-8') as f:
                if hasattr(html_content, 'data'):
                    f.write(html_content.data)
                else:
                    f.write(html_content)
            
            print("\n=== 추출된 기술스택 ===")
            for extraction in result.extractions:
                print(f"분류: {extraction.extraction_class}")
                print(f"기술: {extraction.extraction_text}")
                print(f"속성: {extraction.attributes}")
                print("-" * 40)
            
            print("\n결과가 tech_stack_extraction.jsonl에 저장되었습니다.")
            print("시각화가 tech_stack_visualization.html에 저장되었습니다.")
    else:
        print(f"PDF 파일을 찾을 수 없습니다: {pdf_file_path}")
        print("올바른 PDF 파일 경로를 지정해주세요.")
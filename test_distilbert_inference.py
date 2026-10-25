
from backend.services.distilbert_service import DistilBERTJobScamService


def main():
    print("=" * 70)
    print("DISTILBERT JOB SCAM DETECTION - MULTI-EXAMPLE TEST")
    print("=" * 70)

    service = DistilBERTJobScamService()

    test_jobs = [
        {
            "name": "Example 1 - Professional Job",
            "text": """
            Title: Software Engineer
            Company: Established Technology Company
            Location: Bengaluru, India

            Responsibilities:
            Develop and maintain software applications, collaborate with
            engineering teams, review code, and participate in testing.

            Requirements:
            Bachelor's degree in Computer Science or a related field.
            Experience with Python, Java, or similar programming languages.
            Salary: Competitive, based on experience.
            Apply through the company's official careers portal.
            """,
        },
        {
            "name": "Example 2 - Registration Fee",
            "text": """
            Title: Work From Home Data Entry Assistant

            Earn $5,000 every week working only two hours per day.
            No experience, interview, or qualifications required.
            Pay a registration fee before joining.
            Send the payment immediately to reserve your position.
            """,
        },
        {
            "name": "Example 3 - Unrealistic Salary",
            "text": """
            Title: Online Typing Job

            Earn $1,000 daily with guaranteed income.
            Work for just 15 minutes a day from anywhere.
            Everyone is accepted automatically.
            No interview or skills are needed.
            Contact the recruiter to join immediately.
            """,
        },
        {
            "name": "Example 4 - Internship",
            "text": """
            Title: Machine Learning Intern
            Company: AI Research Team

            Responsibilities:
            Assist with data preprocessing, train machine learning models,
            evaluate model performance, and document experiments.

            Requirements:
            Basic Python knowledge, understanding of machine learning,
            and willingness to learn.

            Duration: Three months.
            Selection: Application review and technical interview.
            """,
        },
        {
            "name": "Example 5 - Suspicious Personal Information Request",
            "text": """
            Title: Immediate Hiring - Remote Assistant

            You are selected without an interview.
            Send your bank account password and one-time verification code
            to complete your recruitment process.
            Pay a processing fee to activate your salary account.
            The position is guaranteed once payment is completed.
            """,
        },
    ]

    for job in test_jobs:
        print("\n" + "-" * 70)
        print(job["name"])
        print("-" * 70)

        try:
            result = service.predict(job["text"])

            print("Prediction:", result["label"])
            print("Fraud probability:", result["fraud_probability"])
            print("Legitimate probability:", result["legitimate_probability"])
            print("Threshold:", result["threshold"])

        except Exception as exc:
            print("ERROR:", exc)

    print("\n" + "=" * 70)
    print("MULTI-EXAMPLE TEST COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()

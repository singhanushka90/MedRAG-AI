from src.evaluation.groq_judge import GroqJudge


def main():

    judge = GroqJudge()

    response = judge.generate(
        "Reply with exactly: Groq judge connection successful."
    )

    print("Judge Model:", judge.get_model_name())
    print("Response:", response)


if __name__ == "__main__":
    main()
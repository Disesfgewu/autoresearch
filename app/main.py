from app.router.LLMClient import LLMClient
from app.util.logger import Logger

def main():
    logger = Logger("Main")
    logger.info("Starting the application...")

    llm_client = LLMClient(logger = logger)
    categories = [
        "Stock",
        "General Purpose",
        "Financial Analysis",
        "Web Search",
        "Event Scheduling"
    ]
    response = llm_client.intentTemplate(
        userInput="What is the stock price of AAPL?",
        categories=categories
    )
    print(response)

if __name__ == "__main__":
    main()

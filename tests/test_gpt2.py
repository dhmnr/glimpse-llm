from glimpse_llm import Glimpse

# Or programmatic mode
# results = glimpse.analyze("The quick brown fox")

def main():
    # Interactive mode
    glimpse = Glimpse("gpt2")
    glimpse.launch()

if __name__ == "__main__":
    main()


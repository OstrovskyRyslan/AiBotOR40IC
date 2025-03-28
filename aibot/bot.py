import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
import openai

# Завдання 1: Лінійна регресія
def linear_regression_example():
    X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1)
    y = np.array([2, 3, 5, 7, 11])  
    
    model = LinearRegression()
    model.fit(X, y)
    
    print("Коефіцієнти:")
    print(f"Нахил (slope): {model.coef_[0]}")
    print(f"Перетин (intercept): {model.intercept_}")
    
    plt.scatter(X, y, color='blue', label='Дані')
    plt.plot(X, model.predict(X), color='red', label='Лінія регресії')
    plt.legend()
    plt.show()

# Завдання 2: Чат-бот OpenAI
def chat_with_openai(prompt):
    client = openai.OpenAI(api_key="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") 

    response = client.chat.completions.create(
        model="gpt-3.5-turbo",
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content

if __name__ == "__main__":
    linear_regression_example()
    
    while True:
        user_input = input("Введіть повідомлення (або 'exit' для виходу): ")
        if user_input.lower() == 'exit':
            break
        reply = chat_with_openai(user_input)
        print("Бот:", reply)

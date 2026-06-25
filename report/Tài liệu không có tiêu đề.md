graph TD
    A[HomeC.csv thô] --> B["Step 1-3: Tiền xử lý và EDA<br/>Member B"]
    B --> C["Step 4: Feature Engineering<br/>Member C"]
    C --> D["Step 5-8: Train và Tune Model<br/>Member A"]
    D --> E[best_model.pkl]
    D --> F[feature_columns.pkl]

    subgraph DashApp [Ứng dụng Plotly Dash]
        G[Giao diện người dùng UI] --> H{Nhập thông số}
        H --> I[Model dự đoán]
        E -.-> I
        F -.-> I
        I -->|Bất thường / Bình thường| G
        I --> J[Gọi API Trợ lý AI]
        J -->|Phân tích và Giải thích| G
    end
import pandas as pd
import pickle
import re

CATEGORY_MAP = {
    1: 'Electronics',
    2: 'Apparel',
    3: 'Books',
    4: 'Home Goods'
}

class CustomerDataExtractor:
    def __init__(self, orders_path, vip_path):
        self.orders_path = orders_path
        self.vip_path = vip_path
        self.vip_ids = set()
        self.raw_data = []

    def load_data(self):
        with open(self.orders_path, 'rb') as f:
            self.raw_data = pickle.load(f)

        with open(self.vip_path, 'r') as f:
            self.vip_ids = {int(line.strip()) for line in f}

    def flatten_data(self):
        rows = []
        for customer in self.raw_data:
            cid_raw = customer.get("id")
            if cid_raw is None:
                continue  # Skip customer if no ID
            cid = int(cid_raw)
            name = customer.get("name", "")

            reg_date = pd.to_datetime(customer.get("registration_date"), errors='coerce')
            if pd.isna(reg_date):
                continue  # Skip customer with invalid registration date

            is_vip = cid in self.vip_ids
            orders = customer.get("orders", [])

            for order in orders:
                oid_raw = order.get("order_id")
                if oid_raw is None:
                    continue  # Skip orders missing order_id
                oid_str = str(oid_raw)
                # Extract numeric part for sorting
                match = re.search(r'\d+', oid_str)
                if match:
                    oid_num = int(match.group())
                else:
                    continue  # Skip if no numeric part found

                odate_raw = order.get("order_date")
                odate = pd.to_datetime(odate_raw, errors='coerce')
                if pd.isna(odate):
                    continue  # Skip orders with invalid date

                items = order.get("items", [])

                total_order = 0
                for i in items:
                    try:
                        price = float(i.get("price", 0))
                    except (TypeError, ValueError):
                        price = 0.0
                    try:
                        quantity = int(i.get("quantity", 0))
                    except (TypeError, ValueError):
                        quantity = 0
                    total_order += price * quantity

                for item in items:
                    pid_raw = item.get("item_id")
                    if pid_raw is None:
                        continue
                    try:
                        pid = int(str(pid_raw).strip())
                    except (TypeError, ValueError):
                        continue

                    pname = item.get("product_name", "")
                    category_code = item.get("category")
                    category = CATEGORY_MAP.get(category_code, "Misc")
                    try:
                        price = float(item.get("price", 0))
                    except (TypeError, ValueError):
                        price = 0.0
                    try:
                        qty = int(item.get("quantity", 0))
                    except (TypeError, ValueError):
                        qty = 0

                    total_price = price * qty
                    percentage = (total_price / total_order * 100) if total_order else 0

                    rows.append({
                        "customer_id": cid,
                        "customer_name": name,
                        "registration_date": reg_date,
                        "is_vip": is_vip,
                        "order_id": oid_str,
                        "order_id_num": oid_num,
                        "order_date": odate,
                        "product_id": pid,
                        "product_name": pname,
                        "category": category,
                        "unit_price": price,
                        "item_quantity": qty,
                        "total_item_price": total_price,
                        "total_order_value_percentage": percentage
                    })

        df = pd.DataFrame(rows)

        # Sort with helper column
        df = df.sort_values(by=["customer_id", "order_id_num", "product_id"], ascending=True)

        # Drop helper
        df = df.drop(columns=["order_id_num"])

        # Final type casting
        df = df.astype({
            "customer_id": "int",
            "customer_name": "str",
            "registration_date": "datetime64[ns]",
            "is_vip": "bool",
            "order_id": "str",
            "order_date": "datetime64[ns]",
            "product_id": "int",
            "product_name": "str",
            "category": "str",
            "unit_price": "float",
            "item_quantity": "int",
            "total_item_price": "float",
            "total_order_value_percentage": "float"
        })

        return df

    def validate_data(self, df):
        print("\nData Validation Report:\n-----------------------")
        # Check missing values
        missing = df.isnull().sum()
        print("Missing values per column:\n", missing)

        # Data types
        print("\nData types:\n", df.dtypes)

        # Negative prices or quantities
        neg_price = df[df['unit_price'] < 0]
        neg_total_price = df[df['total_item_price'] < 0]
        neg_qty = df[df['item_quantity'] < 0]

        print(f"\nNegative unit_price rows: {len(neg_price)}")
        print(f"Negative total_item_price rows: {len(neg_total_price)}")
        print(f"Negative item_quantity rows: {len(neg_qty)}")

        # Orders before registration
        bad_dates = df[df['order_date'] < df['registration_date']]
        print(f"\nOrders before registration date: {len(bad_dates)}")

        # Zero or negative quantity (maybe edge case)
        zero_qty = df[df['item_quantity'] <= 0]
        print(f"\nZero or negative item_quantity rows: {len(zero_qty)}")

        # Duplicate rows
        dup = df.duplicated().sum()
        print(f"\nDuplicate rows in dataset: {dup}")

        # Check totals consistency (order total equals sum of items)
        order_totals = df.groupby(['customer_id', 'order_id'])['total_item_price'].sum()
        percent_totals = df.groupby(['customer_id', 'order_id'])['total_order_value_percentage'].sum()
        inconsistent_orders = order_totals[(percent_totals < 99.9) | (percent_totals > 100.1)]
        print(f"\nOrders with total percentage sum != 100%: {len(inconsistent_orders)}")

        print("\nValidation complete.\n")

if __name__ == "__main__":
    extractor = CustomerDataExtractor(
        "/Users/syuzannaharutyunyan/Desktop/ServiceTitan/customer_orders.pkl",
        "/Users/syuzannaharutyunyan/Desktop/ServiceTitan/vip_customers.txt"
    )
    extractor.load_data()
    df = extractor.flatten_data()
    extractor.validate_data(df)  # Run validation and print report
    df.to_csv("final_customer_orders.csv", index=False)
    print("Extraction complete. CSV saved.")

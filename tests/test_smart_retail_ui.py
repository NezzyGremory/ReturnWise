from __future__ import annotations

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import smart_retail.db as db
from smart_retail.services import create_category, create_transaction, list_products, save_customer, save_product

APP_ENTRY = Path(__file__).resolve().parents[1] / "streamlit_app.py"


@pytest.fixture
def local_db(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(db, "DB_PATH", tmp_path / "smart-retail-ui.sqlite3")
    db.initialize_database()
    return db.DB_PATH


def test_retail_pages_render_without_streamlit_exceptions(local_db):
    pages = [
        "overview.py",
        "pos.py",
        "products.py",
        "inventory.py",
        "transactions.py",
        "customers.py",
        "suppliers.py",
        "reports.py",
        "intelligence.py",
        "settings.py",
    ]
    category_id = create_category("Minuman")
    customer_id = save_customer({"name": "Nadia", "phone": "0812345678"})
    product_id = save_product({
        "sku": "PAGE-001", "name": "Air mineral", "category_id": category_id,
        "cost_price": 500, "selling_price": 1000, "stock": 5, "minimum_stock": 1,
    })
    create_transaction(
        [{"product_id": product_id, "quantity": 1}], payment_method="Tunai",
        amount_paid=1000, customer_id=customer_id,
    )
    app = AppTest.from_file(APP_ENTRY, default_timeout=15).run()
    assert not app.exception

    for page in pages:
        app.switch_page(f"app_pages/smart_retail/{page}").run()
        assert not app.exception, f"Streamlit exception on {page}"


def test_pos_ui_adds_item_and_saves_confirmed_transaction(local_db):
    product_id = save_product({
        "sku": "UI-001",
        "name": "Kopi Susu",
        "cost_price": 700,
        "selling_price": 1200,
        "stock": 3,
        "minimum_stock": 1,
    })
    app = (
        AppTest.from_file(APP_ENTRY, default_timeout=15)
        .run()
        .switch_page("app_pages/smart_retail/pos.py")
        .run()
    )

    app.form_submit_button[0].click().run()
    assert not app.exception
    assert app.session_state["retail_cart"] == {product_id: 1}

    app.number_input(key=f"pos_discount_{app.session_state['retail_checkout_key']}").set_value(200).run()
    assert not app.exception
    assert any(metric.label == "Total pembayaran" and metric.value == "Rp 1.000" for metric in app.metric)

    # AppTest submits form values together; seed the confirmed checkbox in session state
    # before submitting the checkout form to model the same browser interaction.
    app.session_state[f"pos_confirm_{app.session_state['retail_checkout_key']}"] = True
    app.run()
    app.form_submit_button[1].click().run()

    assert not app.exception
    assert list_products()[0]["stock"] == 2
    assert any("tersimpan sebesar Rp 1.000" in item.value for item in app.success)
    assert any(button.label == "Unduh struk untuk dicetak" for button in app.get("download_button"))

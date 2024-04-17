from pydantic import BaseModel, Field
from typing import Annotated, NewType


Intent = Annotated[NewType("Intent", str), "Интент для классификации"]


class IntentModel(BaseModel):
    tag: str
    examples: list[str]


# class ClassResult(BaseModel):
#     tag: Intent
#     result: float = Field(ge=0, le=1)
#
#
# class ClassificationResult(BaseModel):
#     tag: Intent
#     probs: list[ClassResult]


# def some_service_for_crud_data_from_db() -> list[IntentModel]:
#     ...
#
#
# data: list[IntentModel] = some_service_for_crud_data_from_db()

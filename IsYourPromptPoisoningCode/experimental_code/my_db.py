import datetime
import os

from peewee import *

# Database connection details are read from environment variables.
# See `.env.example` at the repository root for the full list of variables.
DB_NAME = os.environ.get("DB_NAME", "prompt_and_code_security")
DB_USER = os.environ.get("DB_USER", "")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "3306"))


def _build_db():
    return MySQLDatabase(
        DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
        host=DB_HOST,
        port=DB_PORT,
    )


db = _build_db()
db_new = _build_db()

class BaseModel(Model):
    class Meta:
        database = db

class BaseModel_new(Model):
    class Meta:
        database = db_new

def db_conf_custom(my_table_name, db_info: MySQLDatabase):
    class BaseModel_custom(Model):
        class Meta:
            database = db_info
    class CodeGenerationRecord(BaseModel_custom):
        """
        Model for storing code generation, modification, and review metadata.
        Each field corresponds to a column in the database table.
        """
        # Primary key field. PeeWee would normally create an INTEGER PRIMARY KEY AUTOINCREMENT field named 'id'.
        # If your 'id' column is BIGINT and you want to specify it explicitly, you can use BigIntegerField.
        id = PrimaryKeyField(primary_key=True)
        pillar_cwe = CharField(max_length=255, null=True, help_text="Abstract CWE ID")

        #
        cwe_id = CharField(max_length=255, null=True, help_text="Common Weakness Enumeration ID")
        task_id = CharField(max_length=255, null=True, help_text="Unique task identifier")
        task_title = TextField(null=True, help_text="Task title")
        prompt_level = CharField(null=True, help_text="Prompt level or difficulty")
        synonym_expansion_id = CharField(max_length=255, null=True, help_text="ID associated with the synonym expansion operation")
        enhancement_id = CharField(max_length=255, null=True, help_text="ID associated with the enhancement operation")
        load_time = DateTimeField(null=True, help_text="Time when the data was loaded or initialized")
        language_version = CharField(max_length=255, null=True, help_text="Prompt language version")
        type = CharField(max_length=255, null=True, help_text="Record type")
        ori_id = CharField(max_length=255, null=True, help_text="ID of the baseline experiment")
        #
        code_model_id = CharField(max_length=255, null=True, help_text="ID of the associated code model")
        code_prompt = TextField(help_text="Original prompt used to generate code")
        generate_ori_text = TextField(null=True, help_text="Original raw model output for code generation")
        generated_code = TextField(help_text="Code generated from the original prompt")
        modified_prompt = TextField(null=True, help_text="Modified or optimized prompt")
        modified_code = TextField(null=True, help_text="New code generated from the modified prompt")
        generation_time = DateTimeField(null=True, help_text="Time when code generation completed")
        #
        judgment_prompt = TextField(null=True, help_text="Prompt used to review or evaluate generated code")
        judgment_result = TextField(null=True, help_text="Review result for the generated code")
        judgment_model_id = CharField(max_length=255, null=True, help_text="ID of the associated review model")
        judgment_flag = CharField(null=True, help_text="Whether a corresponding issue exists: 0 means no issue, 1 means an issue exists")
        judgment_time = DateTimeField(null=True, help_text="Time when the review completed")
        #
        extension = CharField(null=True, help_text="Extension field for storing additional related information")

        class Meta:
            # Define the table name. By default, PeeWee uses the lowercase class name as the table name (e.g. 'codegenerationrecord').
            # If you want to specify a different table name, set it here.
            # table_name = 'your_table_name'
            # table_name = 'json_version'
            table_name = my_table_name

    db.connection()
    db.create_tables([CodeGenerationRecord])
    return CodeGenerationRecord



def db_config(table_name_def):
    # --- Define the data model ---
    class CodeGenerationRecord(BaseModel):
        """
        Model for storing code generation, modification, and review metadata.
        Each field corresponds to a column in the database table.
        """
        # Primary key field. PeeWee would normally create an INTEGER PRIMARY KEY AUTOINCREMENT field named 'id'.
        # If your 'id' column is BIGINT and you want to specify it explicitly, you can use BigIntegerField.
        id = PrimaryKeyField(primary_key=True)
        pillar_cwe = CharField(max_length=255, null=True, help_text="Abstract CWE ID")
        #
        cwe_id = CharField(max_length=255, null=True, help_text="Common Weakness Enumeration ID")
        task_id = CharField(max_length=255, null=True, help_text="Unique task identifier")
        task_title = TextField(null=True, help_text="Task title")
        prompt_level = CharField(null=True, help_text="Prompt level or difficulty")
        synonym_expansion_id = CharField(max_length=255, null=True, help_text="ID associated with the synonym expansion operation")
        enhancement_id = CharField(max_length=255, null=True, help_text="ID associated with the enhancement operation")
        load_time = DateTimeField(null=True, help_text="Time when the data was loaded or initialized")
        language_version = CharField(max_length=255, null=True, help_text="Prompt language version")
        type = CharField(max_length=255, null=True, help_text="Record type")
        ori_id = CharField(max_length=255, null=True, help_text="ID of the baseline experiment")
        #
        code_model_id = CharField(max_length=255, null=True, help_text="ID of the associated code model")
        code_prompt = TextField(help_text="Original prompt used to generate code")
        generate_ori_text = TextField(null=True, help_text="Original raw model output for code generation")
        generated_code = TextField(help_text="Code generated from the original prompt")
        modified_prompt = TextField(null=True, help_text="Modified or optimized prompt")
        modified_code = TextField(null=True, help_text="New code generated from the modified prompt")
        generation_time = DateTimeField(null=True, help_text="Time when code generation completed")
        #
        judgment_prompt = TextField(null=True, help_text="Prompt used to review or evaluate generated code")
        judgment_result = TextField(null=True, help_text="Review result for the generated code")
        judgment_model_id = CharField(max_length=255, null=True, help_text="ID of the associated review model")
        judgment_flag = CharField(null=True, help_text="Whether a corresponding issue exists: 0 means no issue, 1 means an issue exists")
        judgment_time = DateTimeField(null=True, help_text="Time when the review completed")
        #
        extension = CharField(null=True, help_text="Extension field for storing additional related information")

        class Meta:
            # Define the table name. By default, PeeWee uses the lowercase class name as the table name (e.g. 'codegenerationrecord').
            # If you want to specify a different table name, set it here.
            # table_name = 'your_table_name'
            # table_name = 'json_version'
            table_name = table_name_def

    db.connection()
    db.create_tables([CodeGenerationRecord])
    return CodeGenerationRecord


def db_config_new(table_name_def):
    # --- Define the data model ---
    class CodeGenerationRecord(BaseModel_new):
        """
        Model for storing code generation, modification, and review metadata.
        Each field corresponds to a column in the database table.
        """
        # Primary key field. PeeWee would normally create an INTEGER PRIMARY KEY AUTOINCREMENT field named 'id'.
        # If your 'id' column is BIGINT and you want to specify it explicitly, you can use BigIntegerField.
        id = PrimaryKeyField(primary_key=True)
        pillar_cwe = CharField(max_length=255, null=True, help_text="Abstract CWE ID")

        #
        cwe_id = CharField(max_length=255, null=True, help_text="Common Weakness Enumeration ID")
        task_id = CharField(max_length=255, null=True, help_text="Unique task identifier")
        task_title = TextField(null=True, help_text="Task title")
        prompt_level = CharField(null=True, help_text="Prompt level or difficulty")
        synonym_expansion_id = CharField(max_length=255, null=True, help_text="ID associated with the synonym expansion operation")
        enhancement_id = CharField(max_length=255, null=True, help_text="ID associated with the enhancement operation")
        load_time = DateTimeField(null=True, help_text="Time when the data was loaded or initialized")
        language_version = CharField(max_length=255, null=True, help_text="Prompt language version")
        type = CharField(max_length=255, null=True, help_text="Record type")
        ori_id = CharField(max_length=255, null=True, help_text="ID of the baseline experiment")
        #
        code_model_id = CharField(max_length=255, null=True, help_text="ID of the associated code model")
        code_prompt = TextField(help_text="Original prompt used to generate code")
        generate_ori_text = TextField(null=True, help_text="Original raw model output for code generation")
        generated_code = TextField(help_text="Code generated from the original prompt")
        modified_prompt = TextField(null=True, help_text="Modified or optimized prompt")
        modified_code = TextField(null=True, help_text="New code generated from the modified prompt")
        generation_time = DateTimeField(null=True, help_text="Time when code generation completed")
        #
        judgment_prompt = TextField(null=True, help_text="Prompt used to review or evaluate generated code")
        judgment_result = TextField(null=True, help_text="Review result for the generated code")
        judgment_model_id = CharField(max_length=255, null=True, help_text="ID of the associated review model")
        judgment_flag = CharField(null=True, help_text="Whether a corresponding issue exists: 0 means no issue, 1 means an issue exists")
        judgment_time = DateTimeField(null=True, help_text="Time when the review completed")
        #
        extension = CharField(null=True, help_text="Extension field for storing additional related information")

        class Meta:
            # Define the table name. By default, PeeWee uses the lowercase class name as the table name (e.g. 'codegenerationrecord').
            # If you want to specify a different table name, set it here.
            # table_name = 'your_table_name'
            # table_name = 'json_version'
            table_name = table_name_def

    db.connection()
    db.create_tables([CodeGenerationRecord])
    return CodeGenerationRecord


if __name__ == '__main__':

    CodeGenerationRecord = db_config('db_test')
    new1 = CodeGenerationRecord.create(
        cwe_id='CWE-123',
        task_id='task-001',
        prompt_level="0",
        synonym_expansion_id='expansion-001',
        enhancement_id='enhancement-001',
        load_time=datetime.datetime.now(),
    )
    # Insert some example data
    CodeGenerationRecord.create(
        cwe_id='CWE-123',
        task_id='task-001',
        prompt_level='level-A',
        load_time=datetime.datetime.now(),
        generated_code='print("Hello A")'
    )
    CodeGenerationRecord.create(
        cwe_id='CWE-123',
        task_id='task-002',
        prompt_level='level-B',
        load_time=datetime.datetime.now(),
        generated_code='print("Hello B")'
    )
    CodeGenerationRecord.create(
        cwe_id='CWE-456',
        task_id='task-001',
        prompt_level='level-A',
        load_time=datetime.datetime.now(),
        generated_code='print("Hello C")'
    )
    x = CodeGenerationRecord.select().where(
        (CodeGenerationRecord.cwe_id == 'CWE-123') & (CodeGenerationRecord.task_id == 'task-001')
    ).execute()


    y = list(x)
    for i in x:
        print(i.cwe_id)
    for i in y:
        print(i)

    # Modify database data
    new1.prompt_level = 'update_test'
    new1.save()
    db.close()
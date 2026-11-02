import pytest

from watermark_studio.config import ProcessingConfig, processing_route


@pytest.mark.parametrize("mode,scope,first",[
    ("regions","regions","whole"),
    ("whole","full_image","whole"),
    ("focused","full_image","focused"),
    ("thorough","full_image","thorough"),
    ("complete","full_image","complete"),
])
def test_frontend_route_keeps_existing_worker_contract(mode,scope,first):
    assert processing_route(mode)==(scope,first)
    options=ProcessingConfig(scope=scope,first_pass_mode=first,
                             preserve_background=first!="whole")
    assert options.scope==scope and options.first_pass_mode==first


def test_basic_regions_does_not_implicitly_enable_boolean_parameters():
    scope,first=processing_route("regions")
    options=ProcessingConfig(scope=scope,first_pass_mode=first)
    assert not options.preserve_background
    assert not options.quality_check
    assert not options.quality_auto_retry
    with pytest.raises(ValueError,match="未知"):
        processing_route("unexpected")


def test_cli_default_routes_to_automatic_crops_and_keeps_logo_flag_explicit(monkeypatch):
    import sys

    from watermark_studio import cli
    submitted=[]
    def request(url,path,payload=None):
        if path=="/jobs":
            submitted.append(payload)
            return {}
        return {"job":{"state":"completed","completed":1,"total":1,"succeeded":1,
                       "failed":0,"skipped":0,"invalid":0,"quality_counts":{}}}
    monkeypatch.setattr(cli,"request",request)
    monkeypatch.setattr(sys,"argv",["watermark-batch","/mnt/input","/mnt/output","--remove-logo"])
    with pytest.raises(SystemExit) as status:
        cli.main()
    assert status.value.code==0
    options=submitted[0]["options"]
    assert options["scope"]=="regions" and options["first_pass_mode"]=="whole"
    assert options["remove_logo"] is True and options["preserve_background"] is False



@pytest.mark.parametrize("scope,mode,preserve,preset,expected",[
    ("regions","whole",False,"auto",True),
    ("full_image","complete",True,"auto",True),
    ("full_image","whole",True,"auto",True),
    ("full_image","whole",False,"auto",False),
    ("regions","whole",False,"custom",False),
])
def test_original_source_detection_and_physical_vetting_stay_available_for_auto_crops(
        scope,mode,preserve,preset,expected):
    from watermark_studio.config import Region
    from watermark_studio.worker import needs_source_detection

    options=ProcessingConfig(scope=scope,first_pass_mode=mode,preserve_background=preserve,
                             preset=preset,regions=[Region(box=(.1,.1,.5,.5))] if preset=="custom" else [])
    assert needs_source_detection(options) is expected

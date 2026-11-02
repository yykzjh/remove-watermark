import pytest


def test_terminal_loss_uses_exact_clean_not_teacher_endpoint_and_has_gradients():
    torch=pytest.importorskip("torch")
    from watermark_studio.trajectory_training import terminal_objective
    clean=torch.randn(1,12,4);state=torch.randn_like(clean);sigma=.414568
    exact=(state-clean)/sigma
    mask=torch.zeros(1,12);mask[:,:3]=1
    assert terminal_objective(exact,state,clean,sigma,mask).item()<1e-12
    wrong=torch.zeros_like(clean,requires_grad=True)
    loss=terminal_objective(wrong,state,clean,sigma,mask)
    loss.backward()
    assert loss>0 and wrong.grad is not None and wrong.grad.abs().sum()>0
    for bad in [0,float("nan"),True]:
        with pytest.raises(ValueError):terminal_objective(exact,state,clean,bad,mask)


def test_context_buffer_does_not_change_exact_pair_mask_and_noise_matches_pipeline():
    torch=pytest.importorskip("torch")
    from watermark_studio.trajectory_training import context_support, cpu_packed_noise
    values={"edit_mask":torch.zeros(1,64),"source_latents":torch.zeros(1,64,4)}
    values["edit_mask"][0,27]=1;original=values["edit_mask"].clone()
    narrow=context_support(values,128,128,1);wide=context_support(values,128,128,2)
    assert narrow.sum()==9 and wide.sum()==25
    assert torch.equal(original,values["edit_mask"])
    noise=cpu_packed_noise(128,128,4,123,torch.bfloat16)
    expected=torch.randn((1,4,1,8,8),generator=torch.Generator("cpu").manual_seed(123),
        dtype=torch.bfloat16).squeeze(2).flatten(2).transpose(1,2)
    assert torch.equal(noise,expected)
    with pytest.raises(ValueError):context_support(values,128,128,True)


def test_prefix_preservation_uses_full_frame_velocity():
    torch=pytest.importorskip("torch")
    from watermark_studio.trajectory_training import prefix_objective
    teacher=torch.randn(1,10,4)
    assert prefix_objective(teacher,teacher)==0
    altered=teacher.clone();altered[:,-1]+=1
    assert prefix_objective(altered,teacher)>0


def test_terminal_velocity_units_remove_sigma_squared_gradient_attenuation():
    torch=pytest.importorskip("torch")
    from watermark_studio.trajectory_training import terminal_objective
    state=torch.randn(1,12,4);clean=torch.randn_like(state);mask=torch.ones(1,12);sigma=1/12
    prediction=torch.randn_like(state,requires_grad=True)
    ordinary=terminal_objective(prediction,state,clean,sigma,mask)
    normalized=terminal_objective(prediction,state,clean,sigma,mask,normalize_velocity=True)
    expected=(prediction-(state-clean)/sigma).square().mean()
    torch.testing.assert_close(normalized,expected)
    a=torch.autograd.grad(ordinary,prediction,retain_graph=True)[0]
    b=torch.autograd.grad(normalized,prediction)[0]
    torch.testing.assert_close(b,a/(sigma*sigma))


def test_on_policy_prefix_tracks_current_student_and_preserves_source_context_without_graph():
    torch=pytest.importorskip("torch")
    from watermark_studio.trajectory_training import current_prefix, terminal_objective
    noise=torch.randn(1,12,4);clean=torch.randn_like(noise);reference=torch.randn_like(noise)
    support=torch.zeros(1,12,1);support[:,:6]=1;sigmas=[1.,.8,.4,0.]
    calls=[]
    def predict(state,sigma,index):
        calls.append(index);assert not torch.is_grad_enabled()
        return noise-clean
    state=current_prefix(predict,noise,reference,sigmas,support)
    assert calls==[0,1] and not state.requires_grad
    expected=support*(.4*noise+.6*clean)+(1-support)*(.4*noise+.6*reference)
    torch.testing.assert_close(state,expected)
    assert terminal_objective((state-clean)/.4,state,clean,.4,torch.ones(1,12))<1e-12
    shifted=current_prefix(lambda state,sigma,index:noise-clean+1,noise,reference,sigmas,support)
    assert not torch.equal(state[:,:6],shifted[:,:6])
    torch.testing.assert_close(state[:,6:],shifted[:,6:])
    with pytest.raises(ValueError):
        current_prefix(predict,noise,reference,[1.,.4,.8,0.],support)

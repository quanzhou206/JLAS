clc;
clear;
warning off;

range = 400;
Height = 200;
t_all = 10;% 8 An at least 4
A_all = 8;
alpha = 10^(-6);
beta = 10^(-8);
c = 299792458;
L_times = 500;

clock_offset = (2*rand(A_all,1)-1)*alpha;
clock_skew = (2*rand(A_all,1)-1)*beta;

Anchors = [0 0 0;
           range 0 0;
           0 range 0;
           0 0 Height;
           range range 0;
           range 0 Height;
           0 range Height;
           range range Height;]';

x0 = (2*rand(3,1)-1).*[1;1;1]+[200;200;100];

pos = zeros(3,t_all);
pos(:,1) = x0;
v = (2*rand(3,1)-1);
v = 30*v/norm(v);
for t = 2:t_all
    v = (2*rand(3,1)-1);
    v = 30*v/norm(v);
    a = (2*rand(3,1)-1);
    a = 5*a/norm(a);
    pos(:,t) = pos(:,t-1) + v + a/2;
end


figure;
scatter3(Anchors(1,:),Anchors(2,:),Anchors(3,:),200,'p',...
        'MarkerEdgeColor',[0.8500 0.3250 0.0980],...
        'MarkerFaceColor',[0.8500 0.3250 0.0980]);grid on;hold on;
plot3(pos(1,:),pos(2,:),pos(3,:),'linewidth',1.5,'color','#0072BD');
legend('Anchors','Trajectory of Agent','FontSize',10,'interpreter','latex');
xlabel('x (m)','FontSize',12,'interpreter','latex');ylabel('y (m)','FontSize',12,'interpreter','latex');zlabel('z (m)','FontSize',12,'interpreter','latex');

sig_all = [10^(-3),10^(-2.5),10^(-2),10^(-1.5),10^(-1),10^(-0.5),10^(0)];
err_sig_x = zeros(length(sig_all),1);
err_sig_alpha = zeros(length(sig_all),1);
err_sig_beta = zeros(length(sig_all),1);

err_sig_x_CRLB = zeros(length(sig_all),1);
err_sig_alpha_CRLB = zeros(length(sig_all),1);
err_sig_beta_CRLB = zeros(length(sig_all),1);

for sig_now = 1:length(sig_all)
    errx = zeros(L_times,1);
    err_alpha = zeros(L_times,1);
    err_beta = zeros(L_times,1);
    flag_all = zeros(L_times,1);
    for manto = 1:L_times
        %%测距
        range = zeros(A_all,t_all);
        for t=1:t_all
            for an = 1:A_all
                range(an,t) = norm(Anchors(:,an)-pos(:,t)) + c*(clock_offset(an)+clock_skew(an)*(t-1))+normrnd(0,sig_all(sig_now));
            end
        end
        
        diff_x = [diff(pos(1,:));diff(pos(2,:));diff(pos(3,:))];
        theta_hat = [x0;reshape(diff_x,[],1);zeros(2*A_all,1)];
        dtheta = 100*ones(length(theta_hat),1);
        iter = 1;flag = 1;

        tic
        while(norm(dtheta)>1e-3)
            G_all = [];
            b_all = [];
            for local_t = 1:t_all
                [G,b] = Jacobi(theta_hat,Anchors,range(:,local_t),local_t-1,t_all);
                G_all = [G_all;G];
                b_all = [b_all;b];
            end
            dtheta = (G_all'*G_all)\G_all'*b_all;
            theta_hat = theta_hat + dtheta;
            iter = iter+1;
            if iter > 20
                flag = 0;
                break;
            end
        end
        time(manto) = toc;
        flag_all(manto) = flag;
        if flag == 1
            if (sum(isnan(theta_hat))==0)
                errx(manto,1) = norm(theta_hat(1:3)-x0)^2;
                err_alpha(manto,1) = norm(theta_hat(31:38)/c-clock_offset)^2;
                err_beta(manto,1) = norm(theta_hat(39:end)/c-clock_skew)^2;
            else
                flag_all(manto) = 0;
            end
        end
    end
    diff_x = [diff(pos(1,:));diff(pos(2,:));diff(pos(3,:))];
    theta_hat = [x0;reshape(diff_x,[],1);zeros(2*A_all,1)];
    err_sig_x(sig_now) = sqrt(sum(errx)/sum(flag_all));
    err_sig_alpha(sig_now) = sqrt(sum(err_alpha)/sum(flag_all)/A_all);
    err_sig_beta(sig_now) = sqrt(sum(err_beta)/sum(flag_all)/A_all);

    [sigx,sigalpha,sigbeta] = CRLB(theta_hat,Anchors,sig_all(sig_now),t_all);
    err_sig_x_CRLB(sig_now) = sqrt(sigx);
    err_sig_alpha_CRLB(sig_now) = sqrt(sigalpha/A_all)/c;
    err_sig_beta_CRLB(sig_now) = sqrt(sigbeta/A_all)/c;
end


figure;
subplot(3,1,1);
loglog(sig_all,err_sig_x,'-s','linewidth',1.5,'MarkerSize',8);hold on;
loglog(sig_all,err_sig_x_CRLB,'linewidth',1.5,'MarkerSize',8);grid on;
legend('JLAS-TVM (p)','CRLB (p)','location','southeast','FontSize',10,'interpreter','latex');
xlabel('Measurement noise $\sigma (m)$','FontSize',10,'interpreter','latex');
ylabel('RMSE (p) (m)','FontSize',10,'interpreter','latex');

subplot(3,1,2);
loglog(sig_all,err_sig_alpha*c,'-s','linewidth',1.5,'MarkerSize',8);hold on;
loglog(sig_all,err_sig_alpha_CRLB*c,'linewidth',1.5,'MarkerSize',8);grid on;
legend('JLAS-TVM ($\alpha$)','CRLB ($\alpha$)','location','southeast','FontSize',10,'interpreter','latex');
xlabel('Measurement noise $\sigma$ (m)','FontSize',10,'interpreter','latex');
ylabel('RMSE ($\alpha$) (m)','FontSize',10,'interpreter','latex');

subplot(3,1,3);
loglog(sig_all,err_sig_beta*c,'-s','linewidth',1.5,'MarkerSize',8);hold on;
loglog(sig_all,err_sig_beta_CRLB*c,'linewidth',1.5,'MarkerSize',8);grid on;
legend('JLAS-TVM ($\beta$)','CRLB ($\beta$)','location','southeast','FontSize',10,'interpreter','latex');
xlabel('Measurement noise $\sigma$ (m)','FontSize',10,'interpreter','latex');
ylabel('RMSE ($\beta$) (m/s)','FontSize',10,'interpreter','latex');
% ax=gcf;
% ax.Position = [2525.8,40.2,560,450];

function [G,b] = Jacobi(theta,anchors,range,delta,t_all)
    G = zeros(length(range),length(theta));
    b = zeros(length(range),1);
    x = theta(1:3);
    for an = 1:length(range)
        t0 = anchors(:,an) - x;
        for k = 1:delta
            t0 = t0 - theta(3*k+1:3*(k+1));
        end
        G(an,1:3) = - t0/norm(t0);
        for k = 1:delta
            G(an,3*k+1:3*(k+1)) = - t0/norm(t0);
        end
        G(an,3*t_all+an) = 1;
        G(an,3*t_all+an+length(range)) = delta;
        temp = anchors(:,an) - x;
        for k = 1:delta
            temp = temp - theta(3*k+1:3*(k+1));
        end
        b(an) = range(an) - (norm(temp)+theta(3*t_all+an)+delta*theta(3*t_all+an+length(range)));
    end
end

function [sigx,sigalpha,sigbeta] = CRLB(theta,anchors,sig,t_all)
    x = theta(1:3);
    G_all = [];
    for local_t = 1:t_all
        delta = local_t-1;
        G = zeros(size(anchors,2),length(theta));
        for an = 1:size(anchors,2)
            t0 = anchors(:,an) - x;
            for k = 1:delta
                t0 = t0 - theta(3*k+1:3*(k+1));
            end
            G(an,1:3) = - t0/norm(t0);
            for k = 1:delta
                G(an,3*k+1:3*(k+1)) = - t0/norm(t0);
            end
            G(an,3*t_all+an) = 1;
            G(an,3*t_all+an+size(anchors,2)) = delta;
        end
        G_all = [G_all;G];
    end
    Fim = inv(G_all'*G_all/(sig^2));
    sigx = trace(Fim(1:3,1:3));
    sigalpha = trace(Fim(31:38,31:38));
    sigbeta = trace(Fim(39:end,39:end));
end